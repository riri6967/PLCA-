"""Real-traffic scenario: replay ROAD CAN traces on a 10BASE-T1S zone.

Usage (from the project folder, with the virtual environment active):
    python -m traffic.run_road                       # all ROAD ambient logs
    python -m traffic.run_road --log "dataset\\road\\road\\ambient\\ambient_dyno_drive_winter.log"
    python -m traffic.run_road --quick

Each trace is replayed several times with independent time offsets (= the
zone controller aggregates that many CAN segments) and the same four
policies as run_adaptive.py are compared, in two experiments:
  A) CAN only:                 1, 2, 4, 6, 8 CAN segments
  B) CAN + bulk Ethernet (30 % load of large frames, e.g. diagnostics or
     software updates):        1, 2, 3, 4, 5 CAN segments

Where the critical flows cannot be guaranteed, the governor refuses them
(admission control), which is recorded in `road_admission.csv`.

Outputs in `results/`: road_id_stats.csv, road_results.csv, road_summary.csv,
road_admission.csv, road_comparison.png
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plca_sim import PLCAConfig, simulate, summarize  # noqa: E402
from plca_sim.adaptive import AdaptivePolicy, GovernorInfeasible, SafetyGovernor  # noqa: E402
from plca_sim.policy import ConventionalPolicy  # noqa: E402
from plca_sim.road import (  # noqa: E402
    first_lines,
    id_stats,
    make_road_traffic,
    read_candump,
    select_critical_ids,
)

AMBIENT = Path("dataset") / "road" / "road" / "ambient"
POLICIES = ("conventional", "governor", "reactive", "tinyml")
COLORS = {"conventional": "#888888", "governor": "#1f77b4", "reactive": "#ff7f0e", "tinyml": "#2ca02c"}


def replay_log(job: tuple[Path, argparse.Namespace]) -> tuple[list[dict], list[dict], pd.DataFrame | None]:
    """All experiments on one log. Runs in its own process."""
    log, args = job
    name = log.stem.replace("ambient_", "")
    cfg = PLCAConfig(node_count=args.nodes)
    trace = read_candump(log, args.max_trace)
    if trace.empty:
        print(f"Could not read any CAN frames from {log}. First lines of the file:")
        for line in first_lines(log):
            print("   ", line)
        return [], [], None
    if trace.t.max() <= args.duration:
        print(f"{name}: only {trace.t.max():.0f} s of data, skipped")
        return [], [], None

    stats = id_stats(trace)
    stats.insert(0, "log", name)
    model = joblib.load(args.model) if Path(args.model).exists() else None
    policies = POLICIES if model is not None else POLICIES[:3]
    span = trace.t.max()
    crit = select_critical_ids(stats, args.critical_ids, args.critical_skip)
    print(f"{name}: {len(trace) / span:.0f} frames/s, {len(stats)} IDs, critical "
          f"{[hex(c) for c in crit]}", flush=True)

    rows, admission = [], []
    experiments = (("A: CAN only", 0.0, args.copies), ("B: CAN + bulk Ethernet", args.bulk_load, args.bulk_copies))
    for exp, bulk, copies_list in experiments:
        for copies in [int(c) for c in copies_list.split(",")]:
            for seed in range(args.seeds):
                rt = make_road_traffic(trace, cfg, copies, args.duration, seed, args.critical_ids,
                                       args.deadline_ms / 1000, bulk_load=bulk,
                                       critical_skip=args.critical_skip)
                try:
                    gov = SafetyGovernor(cfg, rt.flows, max_frame_bytes=rt.max_frame_bytes)
                    bound_us = max(cfg.bt_to_us(v) for v in gov.latency_bound_bt.values())
                    budget_us = cfg.bt_to_us(gov.nc_budget_bt)
                    admitted = True
                except GovernorInfeasible:
                    bound_us, budget_us, admitted = float("nan"), float("nan"), False
                admission.append({"log": name, "experiment": exp, "copies": copies, "seed": seed,
                                  "offered_load": rt.offered_load, "admitted": admitted,
                                  "flows": len(rt.flows), "bound_us": bound_us, "budget_us": budget_us})
                for pol_name in policies:
                    if pol_name == "conventional":
                        pol = ConventionalPolicy(cfg)
                    elif not admitted:
                        continue
                    else:
                        gov = SafetyGovernor(cfg, rt.flows, max_frame_bytes=rt.max_frame_bytes)
                        mode = {"governor": "off", "reactive": "reactive", "tinyml": "tinyml"}[pol_name]
                        pol = AdaptivePolicy(cfg, gov, model=model, mode=mode)
                    res = simulate(rt.frames, cfg, cfg.s_to_bt(args.duration), pol)
                    row = {"log": name, "experiment": exp, "policy": pol_name, "copies": copies, "seed": seed,
                           "offered_load": rt.offered_load}
                    row.update(summarize(res))
                    rows.append(row)
    return rows, admission, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--log", nargs="*", help="ROAD candump .log file(s) (default: all ambient logs)")
    ap.add_argument("--max-trace", type=float, default=300.0, help="seconds of each log to read")
    ap.add_argument("--copies", default="1,2,4,6,8", help="experiment A: CAN segments per zone")
    ap.add_argument("--bulk-copies", default="1,2,3,4,5", help="experiment B: CAN segments per zone")
    ap.add_argument("--bulk-load", type=float, default=0.3, help="experiment B: bulk Ethernet load")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--critical-ids", type=int, default=2, help="CAN IDs treated as critical")
    ap.add_argument("--critical-skip", type=int, default=0,
                    help="skip this many IDs when picking critical flows (robustness check)")
    ap.add_argument("--deadline-ms", type=float, default=2.0)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--model", default="results/tinyml_model.joblib")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.seeds, args.max_trace = 0.5, 1, 60.0

    logs = [Path(p) for p in args.log] if args.log else sorted(AMBIENT.glob("*.log"))
    missing = [p for p in logs if not p.exists()]
    if missing or not logs:
        sys.exit(f"Log file not found: {missing or AMBIENT}\nPass the right path with --log \"...\"")
    if not Path(args.model).exists():
        print(f"({args.model} not found: run run_tinyml.py first to include the TinyML policy)")
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    t0 = time.time()
    jobs = [(p, args) for p in logs]
    if len(jobs) == 1 or args.workers == 1:
        parts = [replay_log(j) for j in jobs]
    else:
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            parts = pool.map(replay_log, jobs, chunksize=1)

    rows = [r for p in parts for r in p[0]]
    if not rows:
        sys.exit("No results: no log could be replayed.")
    admission = pd.DataFrame([a for p in parts for a in p[1]])
    pd.concat([p[2] for p in parts if p[2] is not None]).to_csv(out / "road_id_stats.csv", index=False)
    admission.to_csv(out / "road_admission.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "road_results.csv", index=False)

    keys = ["experiment", "copies", "policy"]
    means = df.drop(columns=["log", "seed"]).groupby(keys).mean(numeric_only=True)
    means["runs"] = df.groupby(keys).size()
    means["critical_latency_worst_us"] = df.groupby(keys).critical_latency_max_us.max()
    means["critical_miss_worst"] = df.groupby(keys).critical_deadline_miss_rate.max()
    summary = means.reset_index()
    summary.to_csv(out / "road_summary.csv", index=False)

    panels = [
        ("latency_mean_us", "Average latency (µs)"),
        ("latency_p99_us", "99th pct latency (µs)"),
        ("critical_latency_worst_us", "Worst critical latency (µs)"),
        ("critical_deadline_miss_rate", "Critical deadline misses"),
    ]
    policies = [p for p in POLICIES if p in set(summary.policy)]
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
    fig.suptitle(f"Real ROAD CAN traffic replayed on a 10BASE-T1S zone ({len(logs)} log(s))")
    fig.tight_layout()
    fig.savefig(out / "road_comparison.png", dpi=140)
    plt.close(fig)

    cols = ["experiment", "copies", "policy", "runs", "offered_load", "latency_mean_us", "latency_p99_us",
            "critical_latency_worst_us", "critical_deadline_miss_rate", "critical_miss_worst", "loss_rate"]
    print("\n" + summary[cols].round(4).to_string(index=False))
    adm = admission.groupby(["experiment", "copies"]).admitted.mean()
    print("\nShare of runs the governor could guarantee:\n" + adm.round(2).to_string())
    print(f"\nDone in {time.time() - t0:.0f} s. Results saved in '{out}/'.")


if __name__ == "__main__":
    main()
