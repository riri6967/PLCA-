"""Phases 5-6: compare conventional PLCA with the self-adaptive framework.

Usage (from the project folder, with the virtual environment active):
    python -m ml.run_tinyml              # once, to train the TinyML model
    python -m analysis.run_adaptive            # full comparison (a few minutes)
    python -m analysis.run_adaptive --quick    # short run to check everything works

Policies compared on the same traffic (unseen random seeds 200+):
    conventional  standard PLCA, 1 frame per TO
    governor      safety governor only, no adaptation
    reactive      governor + adaptation from the measured current state
    tinyml        governor + adaptation from the TinyML prediction
                  (the proposed self-adaptive PLCA framework)

Outputs in `results/`:
    adaptive_results.csv / adaptive_summary.csv
    adaptive_*.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plca_sim import PLCAConfig, make_traffic, simulate, summarize  # noqa: E402
from plca_sim.adaptive import AdaptivePolicy, SafetyGovernor  # noqa: E402
from plca_sim.policy import ConventionalPolicy  # noqa: E402
from plca_sim.scenarios import ALL_SCENARIOS, critical_flows  # noqa: E402

LOADS = (0.2, 0.3, 0.5, 0.7, 0.9)
POLICIES = ("conventional", "governor", "reactive", "tinyml")
COLORS = {"conventional": "#888888", "governor": "#1f77b4", "reactive": "#ff7f0e", "tinyml": "#2ca02c"}


def load_model(path: Path, cfg: PLCAConfig, quick: bool):
    if path.exists():
        print(f"Using TinyML model {path}")
        return joblib.load(path)
    print(f"{path} not found - training a model first (run_tinyml.py does this properly)...")
    from plca_sim.tinyml import build_dataset, train_tree

    df = build_dataset(ALL_SCENARIOS, (0.1, 0.3, 0.5, 0.7, 0.9), (0, 1) if quick else (0, 1, 2, 3),
                       cfg, 0.3 if quick else 1.0, progress=False)
    model = train_tree(df).tree
    joblib.dump(model, path)
    return model


def make_policy(name: str, scenario: str, cfg: PLCAConfig, model, max_burst: int):
    if name == "conventional":
        return ConventionalPolicy(cfg)
    gov = SafetyGovernor(cfg, critical_flows(scenario))
    mode = {"governor": "off", "reactive": "reactive", "tinyml": "tinyml"}[name]
    return AdaptivePolicy(cfg, gov, model=model, mode=mode, max_burst=max_burst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--duration", type=float, default=2.0, help="simulated seconds per run")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-burst", type=int, default=8, help="max frames per TO when congested")
    ap.add_argument("--model", default="results/tinyml_model.joblib")
    ap.add_argument("--quick", action="store_true", help="0.3 s runs, 1 seed")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.seeds = 0.3, 1

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cfg = PLCAConfig(node_count=args.nodes)
    model = load_model(Path(args.model), cfg, args.quick)

    gov = SafetyGovernor(cfg, critical_flows("mixed"))
    print("\nSafety governor for the mixed-criticality scenario:")
    print(f"  critical frames per TO: {gov.crit_limit}")
    print(f"  non-critical budget between critical TOs: {cfg.bt_to_us(gov.nc_budget_bt):.0f} µs")
    print(f"  guaranteed critical latency bound: "
          f"{max(cfg.bt_to_us(v) for v in gov.latency_bound_bt.values()):.0f} µs (deadline 2000 µs)\n")

    rows = []
    t0 = time.time()
    seeds = range(200, 200 + args.seeds)  # never used for training
    for scenario in ALL_SCENARIOS:
        for load in LOADS:
            for seed in seeds:
                frames = make_traffic(scenario, load, cfg, args.duration, seed)
                for name in POLICIES:
                    pol = make_policy(name, scenario, cfg, model, args.max_burst)
                    res = simulate(frames, cfg, cfg.s_to_bt(args.duration), pol)
                    row = {"policy": name, "scenario": scenario, "load": load, "seed": seed}
                    row.update(summarize(res))
                    row["deferred"] = res.deferred
                    rows.append(row)
            print(f"  {scenario:>13} load {load:.0%} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "adaptive_results.csv", index=False)
    summary = df.drop(columns="seed").groupby(["scenario", "load", "policy"]).mean().reset_index()
    summary.to_csv(out / "adaptive_summary.csv", index=False)

    # Latency / p99 / jitter / throughput per scenario --------------------
    metrics = [
        ("latency_mean_us", "Average latency (µs)"),
        ("latency_p99_us", "99th pct latency (µs)"),
        ("jitter_us", "Jitter (µs)"),
        ("throughput_mbps", "Throughput (Mb/s)"),
    ]
    fig, axes = plt.subplots(len(metrics), len(ALL_SCENARIOS), figsize=(20, 13), squeeze=False)
    for j, sc in enumerate(ALL_SCENARIOS):
        for i, (col, label) in enumerate(metrics):
            ax = axes[i][j]
            for name in POLICIES:
                s = summary[(summary.scenario == sc) & (summary.policy == name)]
                ax.plot(s.load * 100, s[col], marker="o", label=name, color=COLORS[name],
                        lw=2.2 if name == "tinyml" else 1.2)
            if i == 0:
                ax.set_title(sc)
            if j == 0:
                ax.set_ylabel(label)
            if i == len(metrics) - 1:
                ax.set_xlabel("Offered load (%)")
            ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.suptitle("Conventional vs self-adaptive PLCA (unseen test traffic)")
    fig.tight_layout()
    fig.savefig(out / "adaptive_overview.png", dpi=130)
    plt.close(fig)

    # Mixed criticality -------------------------------------------------------
    mixed = summary[summary.scenario == "mixed"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(LOADS))
    w = 0.2
    for k, name in enumerate(POLICIES):
        s = mixed[mixed.policy == name].sort_values("load")
        a1.bar(x + (k - 1.5) * w, s.critical_deadline_miss_rate * 100, w, label=name, color=COLORS[name])
        a2.plot(s.load * 100, s.critical_latency_max_us, marker="o", label=name, color=COLORS[name])
        a3.plot(s.load * 100, s.noncritical_latency_mean_us, marker="o", label=name, color=COLORS[name])
    a1.set_xticks(x, [f"{int(v * 100)}" for v in LOADS])
    a1.set_xlabel("Offered load (%)")
    a1.set_ylabel("Critical deadline misses (%)")
    a1.legend()
    a2.axhline(2000, color="red", ls="--", label="deadline")
    a2.axhline(max(cfg.bt_to_us(v) for v in gov.latency_bound_bt.values()), color="green", ls=":",
               label="governor bound")
    a2.set_yscale("log")
    a2.set_xlabel("Offered load (%)")
    a2.set_ylabel("Worst critical latency (µs)")
    a2.legend(fontsize=8)
    a3.set_xlabel("Offered load (%)")
    a3.set_ylabel("Non-critical average latency (µs)")
    for a in (a1, a2, a3):
        a.grid(alpha=0.3)
    fig.suptitle("Scenario 4: safety-critical traffic")
    fig.tight_layout()
    fig.savefig(out / "adaptive_mixed_criticality.png", dpi=150)
    plt.close(fig)

    # Paper-style figure (poster section 12) --------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("conventional", "tinyml"):
        s = summary[(summary.scenario == "dynamic") & (summary.policy == name)]
        ax.plot(s.load * 100, s.latency_mean_us / 1000, marker="o", color=COLORS[name],
                label="Conventional PLCA" if name == "conventional" else "Adaptive PLCA (TinyML)")
    ax.set_xlabel("Network load (%)")
    ax.set_ylabel("Average latency (ms)")
    ax.set_title("Dynamic traffic: average latency")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "adaptive_latency_vs_load.png", dpi=150)
    plt.close(fig)

    # Text summary -------------------------------------------------------------
    def change(sc, load, col):
        base = summary[(summary.scenario == sc) & (summary.load == load) & (summary.policy == "conventional")][col].iloc[0]
        new = summary[(summary.scenario == sc) & (summary.load == load) & (summary.policy == "tinyml")][col].iloc[0]
        return 100 * (new - base) / base if base else float("nan")

    print("\nAdaptive (tinyml) vs conventional PLCA, change in % (negative = better):")
    table = []
    for sc in ALL_SCENARIOS:
        for load in (0.7, 0.9):
            table.append({
                "scenario": sc, "load": load,
                "mean_latency_%": change(sc, load, "latency_mean_us"),
                "p99_latency_%": change(sc, load, "latency_p99_us"),
                "jitter_%": change(sc, load, "jitter_us"),
                "throughput_%": change(sc, load, "throughput_mbps"),
            })
    print(pd.DataFrame(table).round(1).to_string(index=False))
    m = mixed.set_index(["policy", "load"]).critical_deadline_miss_rate.unstack() * 100
    print("\nMixed criticality - critical deadline misses (%):")
    print(m.round(2).to_string())
    print(f"\nDone in {time.time() - t0:.0f} s. Results saved in '{out}/'.")


if __name__ == "__main__":
    main()
