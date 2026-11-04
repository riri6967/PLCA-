"""Phase 2-3: measure conventional PLCA in every scenario and plot the results.

Usage (from the project folder, with the virtual environment active):
    python run_baseline.py            # full run (a few minutes)
    python run_baseline.py --quick    # short run to check everything works

Outputs go to the `results/` folder:
    results/baseline_results.csv      one row per scenario / load / seed
    results/baseline_summary.csv      averaged over seeds
    results/*.png                     plots
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plca_sim import SCENARIOS, PLCAConfig, make_traffic, simulate, summarize  # noqa: E402

LOADS = (0.2, 0.3, 0.5, 0.7, 0.9)

PLOTS = [
    ("latency_mean_us", "Average latency (µs)"),
    ("latency_p99_us", "99th percentile latency (µs)"),
    ("jitter_us", "Jitter, std. dev. of latency (µs)"),
    ("throughput_mbps", "Throughput (Mb/s)"),
    ("bus_utilization", "Bus utilization"),
    ("to_utilization", "TO utilization (used / total)"),
    ("queue_mean", "Average queue length (frames)"),
    ("loss_rate", "Frame loss rate"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--nodes", type=int, default=8, help="nodes on the bus (default 8)")
    ap.add_argument("--duration", type=float, default=2.0, help="simulated seconds per run")
    ap.add_argument("--seeds", type=int, default=3, help="repetitions per point")
    ap.add_argument("--quick", action="store_true", help="0.2 s runs, 1 seed")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.seeds = 0.2, 1

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cfg = PLCAConfig(node_count=args.nodes)

    rows = []
    t0 = time.time()
    for scenario in SCENARIOS:
        for load in LOADS:
            for seed in range(args.seeds):
                frames = make_traffic(scenario, load, cfg, args.duration, seed)
                res = simulate(frames, cfg, cfg.s_to_bt(args.duration))
                row = {"policy": "conventional", "scenario": scenario, "load": load, "seed": seed}
                row.update(summarize(res))
                rows.append(row)
            print(
                f"{scenario:>13}  load {load:.0%}  "
                f"mean latency {rows[-1]['latency_mean_us']:9.1f} µs  "
                f"TO util {rows[-1]['to_utilization']:.2f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out / "baseline_results.csv", index=False)
    summary = df.drop(columns="seed").groupby(["policy", "scenario", "load"]).mean().reset_index()
    summary.to_csv(out / "baseline_summary.csv", index=False)

    # One overview figure with all KPIs.
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (col, label) in zip(axes.flat, PLOTS):
        for scenario in SCENARIOS:
            s = summary[summary.scenario == scenario]
            ax.plot(s.load * 100, s[col], marker="o", label=scenario)
        ax.set_xlabel("Offered load (%)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(f"Conventional PLCA baseline ({args.nodes} nodes, 10 Mb/s)")
    fig.tight_layout()
    fig.savefig(out / "baseline_overview.png", dpi=150)
    plt.close(fig)

    # Mixed criticality: does the safety-critical flow meet its deadline?
    mixed = summary[summary.scenario == "mixed"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(mixed.load * 100, mixed.critical_latency_mean_us, marker="o", label="critical (mean)")
    a1.plot(mixed.load * 100, mixed.critical_latency_max_us, marker="s", label="critical (max)")
    a1.plot(mixed.load * 100, mixed.noncritical_latency_mean_us, marker="^", label="non-critical (mean)")
    a1.axhline(2000, color="red", ls="--", label="critical deadline (2 ms)")
    a1.set_xlabel("Offered load (%)")
    a1.set_ylabel("Latency (µs)")
    a1.set_yscale("log")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.bar(mixed.load * 100, mixed.critical_deadline_miss_rate * 100, width=6)
    a2.set_xlabel("Offered load (%)")
    a2.set_ylabel("Critical deadline misses (%)")
    a2.grid(alpha=0.3)
    fig.suptitle("Scenario 4: mixed-criticality traffic under conventional PLCA")
    fig.tight_layout()
    fig.savefig(out / "baseline_mixed_criticality.png", dpi=150)
    plt.close(fig)

    print(f"\nDone in {time.time() - t0:.0f} s. Results saved in '{out}/'.")


if __name__ == "__main__":
    main()
