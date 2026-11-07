"""Work Plan 2.2: traffic profiles from the benign CAN traces of CarDS.

Reads the traces directly from dataset/cards/CAN/benign.zip, so they are never
unpacked (5.33 GB). Frames marked T are CarDS's malicious label and are skipped.

Each CAN frame is costed as the Ethernet frame it would become on 10BASE-T1S,
with the same rule as plca_sim/road.py. Loads are per CAN bus, because the
gateway forwards frames between buses: summing buses counts those frames twice.

Outputs in results/cards/:
    cards_traces.csv   one row per trace and bus
    cards_ids.csv      one row per bus and CAN ID, over all traces
    cards_buses.csv    one row per bus

Derived from CarDS (Hellemans et al., ACSAC 2025), CC BY-SA 4.0.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from plca_sim.config import PLCAConfig
from plca_sim.road import frame_bytes, max_in_window

ARCHIVE = Path("dataset/cards/CAN/benign.zip")
# Fig. 1 of the CarDS README. Logs count buses from 0, the figure from 1.
BUSES = {0: "instrument cluster, infotainment", 1: "ABS, brake", 2: "cameras, parking assistant",
         3: "telematics, emergency call", 4: "airbag, motor control unit", 5: "electric powertrain, airbag",
         6: "BMS, AC, inverter", 7: "steering wheel", 8: "doors, steering wheel", 9: "OBD-II port"}
WIRE = {n: PLCAConfig().wire_bits(frame_bytes(n)) for n in range(65)}


def parse(line: bytes) -> tuple[float, int, int, int, bool] | None:
    """(time, bus, CAN ID, payload bytes, CAN FD), or None if not a benign frame."""
    parts = line.split()
    if len(parts) < 3 or (len(parts) > 3 and parts[3] == b"T"):
        return None
    head, _, data = parts[2].partition(b"#")
    fd = data.startswith(b"#")
    if fd:
        data = data[2:]  # second '#' and the flags nibble
    return float(parts[0][1:-1]), int(parts[1][3:]), int(head, 16), len(data) // 2, fd


def profile(name: str) -> tuple[list[dict], list[dict]]:
    times, ids, sizes, fds = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(int)
    with zipfile.ZipFile(ARCHIVE) as z, z.open(name) as fh:
        for line in fh:
            rec = parse(line)
            if rec is None:
                continue
            t, bus, cid, size, fd = rec
            times[bus].append(t)
            ids[bus].append(cid)
            sizes[bus].append(size)
            fds[bus] += fd

    trace = Path(name).stem
    fields = trace.split("-")
    trace_rows, id_rows = [], []
    for bus in sorted(times):
        t = np.array(times[bus])
        cid = np.array(ids[bus])
        n = np.array(sizes[bus])
        span = max(t.max() - t.min(), 1e-9)
        bits = np.vectorize(WIRE.get)(n)
        # Busiest 10 ms of bus time on the T1S wire, in steps of 10 ms.
        slot = ((t - t.min()) / 0.01).astype(np.int64)
        peak = np.bincount(slot, weights=bits).max() / 1e5
        trace_rows.append({"trace": trace, "driving": fields[1], "location": fields[2], "detail": fields[4],
                           "bus": bus, "duration_s": span, "frames": len(t), "frames_per_s": len(t) / span,
                           "ids": len(np.unique(cid)), "fd_share": fds[bus] / len(t), "mean_payload": n.mean(),
                           "t1s_load": bits.sum() / span / 1e7, "t1s_peak_10ms": peak,
                           "max_frames_1ms": max_in_window(t, 1e-3)})
        for c in np.unique(cid):
            ts = t[cid == c]
            gaps = np.diff(ts)
            id_rows.append({"trace": trace, "bus": bus, "can_id": int(c), "frames": len(ts),
                            "duration_s": span, "max_payload": int(n[cid == c].max()),
                            "period_ms": float(np.median(gaps) * 1e3) if len(gaps) else np.nan,
                            "gap_p1_ms": float(np.percentile(gaps, 1) * 1e3) if len(gaps) else np.nan,
                            "gap_p99_ms": float(np.percentile(gaps, 99) * 1e3) if len(gaps) else np.nan,
                            "max_in_2ms": max_in_window(ts, 2e-3), "max_in_10ms": max_in_window(ts, 10e-3)})
    print(trace, f"{sum(len(v) for v in times.values()):,} frames", flush=True)
    return trace_rows, id_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("results/cards"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(ARCHIVE) as z:
        names = [n for n in z.namelist() if n.endswith(".log") and "MACOSX" not in n]
    with mp.Pool(args.workers) as pool:
        done = pool.map(profile, names, chunksize=1)

    traces = pd.DataFrame([r for rows, _ in done for r in rows])
    ids = pd.DataFrame([r for _, rows in done for r in rows])
    traces.to_csv(args.out / "cards_traces.csv", index=False)

    summary = ids.groupby(["bus", "can_id"]).agg(
        frames=("frames", "sum"), traces=("trace", "nunique"), max_payload=("max_payload", "max"),
        period_ms=("period_ms", "median"), gap_p1_ms=("gap_p1_ms", "min"), gap_p99_ms=("gap_p99_ms", "max"),
        max_in_2ms=("max_in_2ms", "max"), max_in_10ms=("max_in_10ms", "max")).reset_index()
    summary.to_csv(args.out / "cards_ids.csv", index=False)

    w = traces.duration_s
    buses = traces.groupby("bus").apply(lambda g: pd.Series({
        "name": BUSES.get(g.name, ""), "frames": g.frames.sum(), "hours": g.duration_s.sum() / 3600,
        "ids": summary[summary.bus == g.name].can_id.nunique(),
        "fd_share": np.average(g.fd_share, weights=g.frames), "mean_payload": np.average(g.mean_payload, weights=g.frames),
        "t1s_load_mean": np.average(g.t1s_load, weights=g.duration_s), "t1s_load_max_trace": g.t1s_load.max(),
        "t1s_peak_10ms": g.t1s_peak_10ms.max(), "max_frames_1ms": g.max_frames_1ms.max()}), include_groups=False).reset_index()
    buses.to_csv(args.out / "cards_buses.csv", index=False)
    print(buses.round(3).to_string(index=False))
    print(f"{len(names)} traces, {w.sum() / 3600:.2f} bus hours, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
