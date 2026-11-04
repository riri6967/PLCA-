"""Replay real in-vehicle CAN traces (ROAD dataset) on a 10BASE-T1S segment.

ROAD ambient logs are in Linux `candump -l` format, e.g.
    (1000000001.000000) can0 0C1#00000000FFFFFFFF
Each line is one CAN frame: timestamp, interface, ID#DATA.

Migration to a zonal 10BASE-T1S segment (assumptions, stated in the paper):
  * Every CAN ID keeps its real timestamps; each CAN frame becomes one
    Ethernet frame (payload <= 8 bytes, so a minimum-size 64-byte frame).
  * ROAD does not say which ECU sends which ID (the DBC is anonymized), so
    IDs are assigned to the N PLCA nodes by balancing their bit load.
  * A zone controller aggregates several CAN segments. "copies" replays the
    trace k times with independent random time offsets (each copy = another
    CAN segment on the same zone), which raises the load while keeping the
    real per-ID timing patterns.
  * Critical flows are chosen by real CAN semantics: among the fastest
    strictly periodic IDs, the ones with the lowest CAN ID, i.e. the highest
    bus priority. Aperiodic IDs such as 0xFFF (lowest CAN priority, gaps down
    to ~1 us) are not eligible even if they are the most frequent.
  * The guarantee does not assume a period: each critical flow declares the
    largest number of frames seen in any deadline-long window of the trace.
  * Optional "bulk" traffic adds native Ethernet data with large frames
    (diagnostics, software updates, sensor streams) on the last nodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PLCAConfig
from .traffic import CriticalFlow, Frame, bursty_traffic

_LINE = re.compile(
    r"^\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s+(\S+)\s+([0-9A-Fa-f]{1,8})#(#?)([0-9A-Fa-fR.]*)"
)

ETH_OVERHEAD = 18  # MAC header (14) + FCS (4)
ETH_MIN = 64


def read_candump(path: str | Path, max_seconds: float | None = None) -> pd.DataFrame:
    """Parse a candump log into columns t (s, from 0), can_id (int), dlc (bytes)."""
    times, ids, dlcs = [], [], []
    t0 = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _LINE.match(line)
            if not m:
                continue
            t = float(m.group(1))
            if t0 is None:
                t0 = t
            if max_seconds is not None and t - t0 > max_seconds:
                break
            data = m.group(5)
            if m.group(4):  # CAN FD "##<flags><data>": first nibble is flags
                data = data[1:]
            dlc = 0 if data.startswith("R") else len(data.replace(".", "")) // 2
            times.append(t - t0)
            ids.append(int(m.group(3), 16))
            dlcs.append(dlc)
    return pd.DataFrame({"t": times, "can_id": ids, "dlc": dlcs})


def first_lines(path: str | Path, n: int = 10) -> list[str]:
    with open(path, "r", errors="replace") as fh:
        return [next(fh, "").rstrip("\n") for _ in range(n)]


def id_stats(trace: pd.DataFrame) -> pd.DataFrame:
    """Per CAN ID: count, rate, median and low-percentile inter-arrival."""
    rows = []
    span = max(trace.t.max(), 1e-9)
    for cid, g in trace.groupby("can_id"):
        gaps = np.diff(g.t.to_numpy())
        rows.append({
            "can_id": cid,
            "count": len(g),
            "rate_hz": len(g) / span,
            "period_median_s": float(np.median(gaps)) if len(gaps) else np.nan,
            "period_p1_s": float(np.percentile(gaps, 1)) if len(gaps) else np.nan,
            "dlc": int(g.dlc.max()),
        })
    return pd.DataFrame(rows).sort_values("rate_hz", ascending=False).reset_index(drop=True)


def select_critical_ids(stats: pd.DataFrame, n: int) -> list[int]:
    """Highest-priority (lowest) CAN IDs among the fastest periodic IDs."""
    if n <= 0:
        return []
    periodic = stats[stats.period_p1_s >= 0.5 * stats.period_median_s]
    if periodic.empty:
        raise ValueError("no periodic CAN ID in this trace: cannot pick critical flows")
    fastest = periodic.period_median_s.min()
    near = periodic.period_median_s <= 1.25 * fastest
    first = periodic[near].sort_values("can_id")
    rest = periodic[~near].sort_values(["period_median_s", "can_id"])
    return [int(c) for c in pd.concat([first, rest]).can_id.head(n)]


def max_in_window(times: np.ndarray, window_s: float) -> int:
    """Largest number of events in any half-open window of length `window_s`."""
    t = np.sort(np.asarray(times))
    if len(t) == 0:
        return 0
    return int(np.max(np.searchsorted(t, t + window_s, side="left") - np.arange(len(t))))


def frame_bytes(dlc: int) -> int:
    return max(ETH_MIN, ETH_OVERHEAD + dlc)


@dataclass
class RoadTraffic:
    frames: list[Frame]
    flows: tuple[CriticalFlow, ...]
    offered_load: float
    id_to_node: dict[tuple[int, int], int]

    @property
    def max_frame_bytes(self) -> int:
        return max((f.size for f in self.frames), default=ETH_MIN)


def make_road_traffic(
    trace: pd.DataFrame,
    cfg: PLCAConfig,
    copies: int,
    duration_s: float,
    seed: int,
    n_critical: int = 2,
    critical_deadline_s: float = 2e-3,
    bulk_load: float = 0.0,
    bulk_nodes: int = 2,
) -> RoadTraffic:
    """Turn `copies` time-shifted replays of a CAN trace into PLCA traffic."""
    rng = np.random.default_rng(seed)
    stats = id_stats(trace)
    span = float(trace.t.max())
    if span < duration_s:
        raise ValueError(f"trace is only {span:.1f} s long; use a shorter duration")
    crit_ids = set(select_critical_ids(stats, n_critical))

    # Assign (copy, CAN ID) pairs to nodes, heaviest first, to the least loaded node.
    load = np.zeros(cfg.node_count)
    id_to_node: dict[tuple[int, int], int] = {}

    def bits_of(r) -> float:
        return r.rate_hz * cfg.wire_bits(frame_bytes(int(r.dlc)))

    # Critical flows first, spread evenly: one node per flow where possible,
    # which leaves the governor the largest non-critical budget.
    plain = [n for n in range(cfg.node_count)
             if not (bulk_load > 0 and n >= cfg.node_count - bulk_nodes)]
    bulky = [n for n in range(cfg.node_count) if n not in plain]
    order = plain + bulky
    k = 0
    for c in range(copies):
        for _, r in stats[stats.can_id.isin(crit_ids)].iterrows():
            node = order[k % len(order)]
            k += 1
            id_to_node[(c, int(r.can_id))] = node
            load[node] += bits_of(r)
    items = []
    for c in range(copies):
        for _, r in stats[~stats.can_id.isin(crit_ids)].iterrows():
            items.append((bits_of(r), c, int(r.can_id)))
    items.sort(reverse=True)
    for bits, c, cid in items:
        node = int(np.argmin(load))
        id_to_node[(c, cid)] = node
        load[node] += bits

    t_all = trace.t.to_numpy()
    ids_all = trace.can_id.to_numpy()
    dlc_all = trace.dlc.to_numpy()
    deadline_bt = cfg.s_to_bt(critical_deadline_s)
    horizon = cfg.s_to_bt(duration_s)
    frames: list[Frame] = []
    for c in range(copies):
        off = float(rng.uniform(0, span - duration_s))
        sel = (t_all >= off) & (t_all < off + duration_s)
        for t, cid, dlc in zip(t_all[sel], ids_all[sel], dlc_all[sel]):
            crit = int(cid) in crit_ids
            arrival = cfg.s_to_bt(float(t) - off)
            if arrival < horizon:
                frames.append(Frame(id_to_node[(c, int(cid))], arrival, frame_bytes(int(dlc)),
                                    crit, deadline_bt if crit else None))

    if bulk_load > 0:
        sizes, probs = (512, 1024, 1518), (0.2, 0.3, 0.5)
        for node in range(cfg.node_count - bulk_nodes, cfg.node_count):
            frames += bursty_traffic(rng, cfg, node, bulk_load / bulk_nodes, horizon,
                                     8.0, sizes, probs)

    # Critical flows for the governor. Real traces jitter, so the guarantee
    # uses the worst count measured over the whole trace, not a period.
    flows = []
    for _, r in stats[stats.can_id.isin(crit_ids)].iterrows():
        worst = max_in_window(t_all[ids_all == int(r.can_id)], critical_deadline_s)
        for c in range(copies):
            flows.append(CriticalFlow(id_to_node[(c, int(r.can_id))], float(r.period_median_s),
                                      frame_bytes(int(r.dlc)), critical_deadline_s, worst))

    offered = sum(cfg.wire_bits(f.size) for f in frames) / horizon
    return RoadTraffic(frames, tuple(flows), offered, id_to_node)
