"""Work Plan 5.2: figures and tables from the OMNeT++ results.

    python -m analysis.plot_omnet

Reads results/omnet/summary.csv and model_comparison.csv. Writes PNG figures and
the tables behind them to results/omnet/figures/. Means are over seeds, with
95% confidence intervals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plca_sim.monitor import FEATURES  # noqa: E402

ROOT = Path("results/omnet")
OUT = ROOT / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"B1": "#2a78d6", "B2": "#eb6834", "B3": "#1baf7a", "B4": "#eda100", "P": "#6250d6", "A1": "#e87ba4",
          "A2": "#008300", "A3": "#52514e", "S1/S2": "#2a78d6", "S3": "#eb6834", "S4": "#1baf7a"}
MARKERS = {"B1": "o", "B2": "s", "B3": "^", "B4": "D", "P": "P", "S1/S2": "o", "S3": "s", "S4": "^"}
T95 = {3: 4.303, 10: 2.262}
NAMES = {"S4_Mixed": "B1", "B3_20": "B3", "B4_20": "B4", "P20": "P", "A1_20": "A1", "A2_20": "A2", "A3_20": "A3"}
CONFIG = {v: k for k, v in NAMES.items()}
LABELS = {"B1": "B1 conventional", "B2": "B2 best static", "B3": "B3 fixed priority", "B4": "B4 reactive EWMA",
          "P": "P proposed", "A1": "A1 no governor", "A2": "A2 no reordering", "A3": "A3 wrong predictions"}


def stats(df: pd.DataFrame, column: str, by: str = "load") -> pd.DataFrame:
    g = df.groupby(by)[column].agg(["mean", "std", "count"]).reset_index()
    g["ci"] = [T95.get(int(n), 2.0) * s / np.sqrt(n) if n > 1 else 0.0 for s, n in zip(g["std"].fillna(0), g["count"])]
    return g


def axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def line(ax, s: pd.DataFrame, key: str, label: str, x: str = "load") -> None:
    ax.errorbar(s[x], s["mean"], yerr=s["ci"], color=COLORS[key], marker=MARKERS.get(key, "o"), markersize=7,
                linewidth=2, capsize=3, label=label, markeredgecolor="white", markeredgewidth=1)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def best_static(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """B2: per load, the static setting with the fewest misses, then the lowest p99."""
    run = df[df.config == "B2_Tuned"].copy()
    run["setting"] = run.run.str.extract(r"(bs=\d+.*?bg=\d+)")[0]
    keep = []
    for load, g in run.groupby("load"):
        score = g.groupby("setting")[[column, "lat_p99_ms"]].mean().sort_values([column, "lat_p99_ms"])
        keep.append(g[g.setting == score.index[0]])
    return pd.concat(keep) if keep else run


def latency_vs_load(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for config, key in (("S1_S2_Uniform", "S1/S2"), ("S3_Bursty", "S3"), ("S4_Mixed", "S4")):
        line(ax, stats(df[df.config == config], "lat_p99_ms"), key, {"S1/S2": "S1 and S2 uniform", "S3": "S3 bursty", "S4": "S4 mixed"}[key])
    ax.set_yscale("log")
    axes(ax, "Offered load", "99th percentile latency (ms)")
    ax.legend(frameon=False)
    save(fig, "latency_vs_load")


def safety_miss(df: pd.DataFrame) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    rows = []
    b2 = best_static(df, "safety_miss_20ms")
    for key in ("B1", "B2", "B3", "B4", "P"):
        part = b2 if key == "B2" else df[df.config == CONFIG[key]]
        if part.empty:
            continue
        s = stats(part, "safety_miss_20ms")
        s[["mean", "ci"]] *= 100
        line(ax, s, key, LABELS[key])
        rows.append(s.assign(design=key)[["design", "load", "mean", "ci"]])
    axes(ax, "Offered load", "Safety frames past 20 ms or dropped (%)")
    ax.legend(frameon=False)
    save(fig, "safety_miss_ratio")
    table = pd.concat(rows) if rows else pd.DataFrame()
    table.to_csv(OUT / "safety_miss_ratio.csv", index=False)
    return table


def scalability(df: pd.DataFrame) -> None:
    run = df[df.config.isin(["S2_Scale", "S1_S2_Uniform"]) & df.load.isin([0.5, 0.9])]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
    for load, key in ((0.5, "S3"), (0.9, "S1/S2")):
        part = run[run.load == load]
        line(left, stats(part, "lat_p99_ms", "numNodes"), key, f"Load {load}", "numNodes")
        line(right, stats(part, "cycle_max_ms", "numNodes"), key, f"Load {load}", "numNodes")
    axes(left, "Followers", "99th percentile latency (ms)")
    axes(right, "Followers", "Longest PLCA cycle (ms)")
    left.set_xticks([4, 8, 12, 16])
    right.set_xticks([4, 8, 12, 16])
    left.legend(frameon=False)
    save(fig, "scalability")


def governor(df: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
    for config, key, name in (("P20", "P", "P proposed"), ("B4_20", "B4", "B4 reactive EWMA")):
        part = df[df.config == config].copy()
        if part.empty:
            continue
        part["override_pct"] = part.override_rate * 100
        part["changes_per_s"] = part.changes / 55
        line(left, stats(part, "override_pct"), key, name)
        line(right, stats(part, "changes_per_s"), key, name)
    axes(left, "Offered load", "Proposals refused by the governor (%)")
    axes(right, "Offered load", "Accepted reconfigurations per second")
    left.legend(frameon=False)
    save(fig, "override_rate")


def row(part: pd.DataFrame, design: str) -> dict:
    # Every design here is judged at its own deadline except B1 and B2, which have none: use 20 ms.
    column = "safety_miss_20ms" if part.config.iloc[0] in ("S4_Mixed", "B2_Tuned") else "safety_miss_rate"
    miss = stats(part.assign(pct=part[column] * 100), "pct", "config").iloc[0]
    p99 = stats(part, "lat_p99_ms", "config").iloc[0]
    return {"design": design, "runs": len(part), "miss_pct": miss["mean"], "miss_ci": miss["ci"],
            "safety_worst_ms": part.safety_max_ms.max(), "p99_ms": p99["mean"], "p99_ci": p99["ci"],
            "mean_ms": part.lat_mean_ms.mean(), "override_pct": part.override_rate.mean() * 100,
            "changes_per_s": part.changes.mean() / 55}


def tables(df: pd.DataFrame) -> None:
    """Work Plan 5.3: ablation and sensitivity at load 0.9, one CSV each."""
    high = df[df.load == 0.9]
    groups = {"designs": ("S4_Mixed", "B2_Tuned", "B3_20", "B4_20", "P20"),
              "ablation": ("P20", "A1_20", "A2_20", "A3_20"),
              "sensitivity_model": ("P20", "M_20", "B4_20"),
              "sensitivity_window": ("W2_20", "P20", "W20_20"),
              "sensitivity_deadline": ("D15_Deadline", "P20", "D30_Deadline"),
              "bursty": ("S5_MixedBursty", "S5_B3_20", "S5_B4_20", "S5_P20")}
    for name, configs in groups.items():
        rows = [row(best_static(high, "safety_miss_20ms") if c == "B2_Tuned" else high[high.config == c], c)
                for c in configs]
        if rows:
            pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
            print(name, pd.DataFrame(rows).round(3).to_string(index=False), sep=chr(10))


def paired(df: pd.DataFrame) -> None:
    """Same-seed differences, because seeds share traffic. Negative means the first design is better."""
    rows = []
    for one, other in (("P20", "B3_20"), ("P20", "B4_20"), ("S5_P20", "S5_B3_20"), ("S5_P20", "S5_B4_20")):
        for load in sorted(df[df.config == one].load.unique()):
            a = df[(df.config == one) & (df.load == load)].set_index("seed")
            b = df[(df.config == other) & (df.load == load)].set_index("seed")
            for column in ("lat_mean_ms", "lat_p99_ms"):
                d = (a[column] - b[column]).dropna()
                half = T95[10] * d.std() / np.sqrt(len(d))
                rows.append({"pair": f"{one} - {other}", "load": load, "metric": column, "diff": d.mean(), "ci": half,
                             "pct": 100 * d.mean() / b[column].mean(), "significant": abs(d.mean()) > half})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "paired_differences.csv", index=False)
    print(table.round(3).to_string(index=False))


def timeline() -> None:
    """Queue and utilization around bursts, one 10 s run each at load 0.9."""
    names = ["time_s", *FEATURES]
    paths = {"Conventional": ROOT / "timeline/T_Conventional-3.csv", "Proposed": ROOT / "timeline/T_Proposed-3.csv"}
    if not all(p.exists() for p in paths.values()):
        return
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for (name, path), color in zip(paths.items(), (COLORS["B1"], COLORS["P"])):
        d = pd.read_csv(path, header=None, names=names)
        d = d[(d.time_s >= 4) & (d.time_s < 6)]
        ax.plot(d.time_s, d.queue_total, color=color, linewidth=1.2, label=name)
    axes(ax, "Simulated time (s)", "Frames queued on the segment")
    ax.legend(frameon=False)
    save(fig, "timeline_bursts")


def models() -> None:
    table = pd.read_csv(ROOT / "model_comparison.csv").sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = np.arange(len(table))
    ax.barh(y, table.macro_f1, color=COLORS["B1"], height=0.6)
    ax.set_yticks(y, table.model)
    for yi, v in zip(y, table.macro_f1):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Macro F1 on the unseen seed", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.set_xticks([])
    save(fig, "model_comparison")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "summary.csv")
    latency_vs_load(df)
    print(safety_miss(df).round(3).to_string(index=False))
    scalability(df)
    governor(df)
    tables(df)
    paired(df)
    timeline()
    models()


if __name__ == "__main__":
    main()
