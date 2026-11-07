"""Replay CarDS driving traces on a 10BASE-T1S zone, as run_road.py does for ROAD.

A zone merges the CAN buses of one vehicle function. The gateway forwards some
IDs between buses, so an ID seen on several buses of a zone is kept only on the
bus where it is most frequent. CarDS publishes no signal definitions, so the
critical flows are an assumption: the fastest periodic IDs, as with ROAD.

Outputs in results/cards/: cards_replay.csv, cards_replay_summary.csv

Derived from CarDS (Hellemans et al., ACSAC 2025), CC BY-SA 4.0.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import zipfile
from pathlib import Path

import pandas as pd

from plca_sim import PLCAConfig, simulate, summarize
from plca_sim.adaptive import AdaptivePolicy, GovernorInfeasible, SafetyGovernor
from plca_sim.policy import ConventionalPolicy
from plca_sim.road import make_road_traffic
from traffic.run_cards_profile import ARCHIVE, parse

ZONES = {"body": (7, 8), "chassis": (1, 4), "powertrain": (5, 6), "cockpit": (0, 2, 3)}
TRACES = ("FastDriving-DrivingSchool-NA-NA", "ReverseDriving-DrivingSchool-NA-NA", "ReverseDriving-University-NA-NA_1",
          "ReverseDriving-University-NA-NA_2", "SlowDriving-DrivingSchool-NA-NA", "SlowDriving-University-NA-NA_1",
          "SlowDriving-University-NA-NA_2", "Standstill-University-NA-NA_1", "Standstill-University-NA-NA_2")


def read_zone(trace: str, buses: tuple[int, ...], max_s: float) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(ARCHIVE) as z, z.open(f"benign/Benign-{trace}.log") as fh:
        for line in fh:
            rec = parse(line)
            if rec is None:
                continue
            if not rows:
                start = rec[0]
            if rec[0] - start > max_s:
                break
            if rec[1] in buses:
                rows.append((rec[0] - start, rec[1], rec[2], rec[3]))
    df = pd.DataFrame(rows, columns=["t", "bus", "can_id", "dlc"])
    home = df.groupby(["can_id", "bus"]).size().reset_index(name="n").sort_values("n").drop_duplicates("can_id", keep="last")
    df = df.merge(home[["can_id", "bus"]], on=["can_id", "bus"])
    return df.sort_values("t").reset_index(drop=True)[["t", "can_id", "dlc"]]


def replay(job: tuple[str, str, argparse.Namespace]) -> list[dict]:
    trace_name, zone, args = job
    cfg = PLCAConfig(node_count=args.nodes)
    trace = read_zone(trace_name, ZONES[zone], args.max_trace)
    rows = []
    for deadline_ms in (2.0, 5.0, 10.0):
        for bulk in (0.0, 0.3):
            for seed in range(args.seeds):
                rt = make_road_traffic(trace, cfg, 1, args.duration, seed, 2, deadline_ms / 1000, bulk_load=bulk)
                base = {"trace": trace_name, "zone": zone, "deadline_ms": deadline_ms, "bulk": bulk, "seed": seed,
                        "offered_load": rt.offered_load, "max_frame": rt.max_frame_bytes}
                res = simulate(rt.frames, cfg, cfg.s_to_bt(args.duration), ConventionalPolicy(cfg))
                rows.append({**base, "policy": "conventional", "admitted": True, **summarize(res)})
                try:
                    gov = SafetyGovernor(cfg, rt.flows, max_frame_bytes=rt.max_frame_bytes)
                except GovernorInfeasible:
                    rows.append({**base, "policy": "governor", "admitted": False})
                    continue
                res = simulate(rt.frames, cfg, cfg.s_to_bt(args.duration), AdaptivePolicy(cfg, gov, model=None, mode="off"))
                rows.append({**base, "policy": "governor", "admitted": True, **summarize(res)})
    print(trace_name, zone, len(trace), "frames", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--max-trace", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("results/cards"))
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(t, z, args) for t in TRACES for z in ZONES]
    with mp.Pool(args.workers) as pool:
        df = pd.DataFrame([r for rows in pool.map(replay, jobs, chunksize=1) for r in rows])
    df.to_csv(args.out / "cards_replay.csv", index=False)
    ok = df[df.admitted.astype(bool)]
    keys = ["zone", "bulk", "deadline_ms", "policy"]
    out = ok.groupby(keys).agg(runs=("seed", "size"), load=("offered_load", "mean"),
                               miss_mean=("critical_deadline_miss_rate", "mean"),
                               runs_with_miss=("critical_deadline_miss_rate", lambda s: int((s > 0).sum())),
                               worst_ms=("critical_latency_max_us", lambda s: s.max() / 1e3)).reset_index()
    refused = df[~df.admitted.astype(bool)].groupby(keys).size().rename("refused").reset_index()
    out = out.merge(refused, how="left", on=keys).fillna({"refused": 0})
    out.to_csv(args.out / "cards_replay_summary.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
