"""Real-traffic scenario: replay ROAD CAN traces on a 10BASE-T1S zone.

Usage (from the project folder, with the virtual environment active):
    python run_road.py                       # default ROAD highway log
    python run_road.py --log "dataset\\road\\road\\ambient\\ambient_dyno_drive_winter.log"
    python run_road.py --quick

The trace is replayed several times with independent time offsets (= the
zone controller aggregates that many CAN segments) and the same four
policies as run_adaptive.py are compared, in two experiments:
  A) CAN only:                 1, 2, 4, 6, 8 CAN segments
  B) CAN + bulk Ethernet (30 % load of large frames, e.g. diagnostics or
     software updates):        1, 2, 3, 4 CAN segments

Outputs in `results/`: road_id_stats.csv, road_results.csv, road_summary.csv,
road_*.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plca_sim import PLCAConfig, simulate, summarize  # noqa: E402
from plca_sim.adaptive import AdaptivePolicy, GovernorInfeasible, SafetyGovernor  # noqa: E402
from plca_sim.policy import ConventionalPolicy  # noqa: E402
from plca_sim.road import first_lines, id_stats, make_road_traffic, read_candump  # noqa: E402

DEFAULT_LOG = Path("dataset") / "road" / "road" / "ambient" / "ambient_highway_street_driving_long.log"
POLICIES = ("conventional", "governor", "reactive", "tinyml")
COLORS = {"conventional": "#888888", "governor": "#1f77b4", "reactive": "#ff7f0e", "tinyml": "#2ca02c"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="ROAD candump .log file")
    ap.add_argument("--max-trace", type=float, default=300.0, help="seconds of the log to read")
    ap.add_argument("--copies", default="1,2,4,6,8", help="experiment A: CAN segments per zone")
    ap.add_argument("--bulk-copies", default="1,2,3,4", help="experiment B: CAN segments per zone")
    ap.add_argument("--bulk-load", type=float, default=0.3, help="experiment B: bulk Ethernet load")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--critical-ids", type=int, default=2, help="fastest IDs treated as critical")
    ap.add_argument("--deadline-ms", type=float, default=2.0)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--model", default="results/tinyml_model.joblib")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.seeds, args.max_trace = 0.5, 1, 60.0

    log = Path(args.log)
    if not log.exists():
        sys.exit(f"Log file not found: {log}\nPass the right path with --log \"...\"")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cfg = PLCAConfig(node_count=args.nodes)

    print(f"Reading {log} (first {args.max_trace:.0f} s)...")
    trace = read_candump(log, args.max_trace)
    if trace.empty:
        print("Could not read any CAN frames. First lines of the file:")
        for line in first_lines(log):
            print("   ", line)
        sys.exit("Please paste these lines to Claude so the reader can be adapted.")

    stats = id_stats(trace)
    stats.to_csv(out / "road_id_stats.csv", index=False)
    span = trace.t.max()
    can_load = (trace.dlc * 8 + 47).sum() / span / 500e3  # approx. CAN bits, 500 kb/s
    print(f"  {len(trace)} CAN frames, {len(stats)} IDs, {span:.1f} s, "
          f"{len(trace) / span:.0f} frames/s (~{can_load:.0%} of a 500 kb/s CAN bus)")
    print("  fastest IDs:")
    for _, r in stats.head(5).iterrows():
        print(f"    0x{int(r.can_id):03X}: {r.rate_hz:7.1f} Hz, period ~{r.period_median_s * 1000:.1f} ms")

    import joblib

    model = joblib.load(args.model) if Path(args.model).exists() else None
    policies = POLICIES if model is not None else POLICIES[:3]
    if model is None:
        print(f"  ({args.model} not found: run run_tinyml.py first to include the TinyML policy)")

    rows = []
    t0 = time.time()
    experiments = [("A: CAN only", 0.0, args.copies), ("B: CAN + bulk Ethernet", args.bulk_load, args.bulk_copies)]
    for exp, bulk, copies_list in experiments:
        print(f"\nExperiment {exp}")
        for copies in [int(c) for c in copies_list.split(",")]:
            for seed in range(args.seeds):
                rt = make_road_traffic(trace, cfg, copies, args.duration, seed,
                                       args.critical_ids, args.deadline_ms / 1000, bulk_load=bulk)
                for name in policies:
                    if name == "conventional":
                        pol = ConventionalPolicy(cfg)
                    else:
                        try:
                            gov = SafetyGovernor(cfg, rt.flows, max_frame_bytes=rt.max_frame_bytes)
                        except GovernorInfeasible as e:
                            if seed == 0:
                                print(f"  {copies} segment(s): no guarantee possible ({e})")
                            break
                        mode = {"governor": "off", "reactive": "reactive", "tinyml": "tinyml"}[name]
                        pol = AdaptivePolicy(cfg, gov, model=model, mode=mode)
                    res = simulate(rt.frames, cfg, cfg.s_to_bt(args.duration), pol)
                    row = {"experiment": exp, "policy": name, "copies": copies, "seed": seed,
                           "offered_load": rt.offered_load}
                    row.update(summarize(res))
                    rows.append(row)
            print(f"  {copies} CAN segment(s): offered load {rt.offered_load:.0%} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "road_results.csv", index=False)
    summary = df.drop(columns="seed").groupby(["experiment", "copies", "policy"]).mean().reset_index()
    summary.to_csv(out / "road_summary.csv", index=False)

    panels = [
        ("latency_mean_us", "Average latency (µs)"),
        ("latency_p99_us", "99th pct latency (µs)"),
        ("critical_latency_max_us", "Worst critical latency (µs)"),
        ("critical_deadline_miss_rate", "Critical deadline misses"),
    ]
    exps = list(summary.experiment.unique())
    fig, axes = plt.subplots(len(exps), 4, figsize=(19, 4.2 * len(exps)), squeeze=False)
    for i, exp in enumerate(exps):
        for ax, (col, label) in zip(axes[i], panels):
            for name in policies:
                s = summary[(summary.experiment == exp) & (summary.policy == name)]
                ax.plot(s.offered_load * 100, s[col], marker="o", label=name, color=COLORS[name])
            ax.set_xlabel("Offered load on 10BASE-T1S (%)")
            ax.set_title(f"{exp}: {label}", fontsize=10)
            ax.grid(alpha=0.3)
        axes[i][2].axhline(args.deadline_ms * 1000, color="red", ls="--", label="deadline")
        axes[i][0].legend()
    fig.suptitle(f"Real ROAD CAN traffic replayed on a 10BASE-T1S zone ({log.name})")
    fig.tight_layout()
    fig.savefig(out / "road_comparison.png", dpi=140)
    plt.close(fig)

    cols = ["experiment", "copies", "policy", "offered_load", "latency_mean_us", "latency_p99_us",
            "critical_latency_max_us", "critical_deadline_miss_rate", "loss_rate"]
    print("\n" + summary[cols].round(3).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f} s. Results saved in '{out}/'.")


if __name__ == "__main__":
    main()
