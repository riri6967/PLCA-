"""What does the guarantee cost, and does adapting the budget help?

Usage (from the project folder, with the virtual environment active):
    python run_tradeoff.py            # after run_tinyml.py
    python run_tradeoff.py --quick

Three studies on the mixed-criticality scenario:
  1. deadline  how much non-critical latency the guarantee costs as the
               critical deadline is relaxed (1.8-10 ms)
  2. budget    whether adapting the non-critical budget to the predicted
               state helps (the budget can only shrink: more would break
               the proof)
  3. topology  sensitivity to node count and TO timer, and the check of the
               analytical bound against the worst simulated latency

Outputs in `results/`: tradeoff_deadline.csv, tradeoff_budget.csv,
tradeoff_topology.csv, tradeoff_deadline.png
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plca_sim import PLCAConfig, make_traffic, simulate, summarize  # noqa: E402
from plca_sim.adaptive import AdaptivePolicy, GovernorInfeasible, SafetyGovernor  # noqa: E402
from plca_sim.policy import ConventionalPolicy  # noqa: E402
from plca_sim.scenarios import CRITICAL_NODES, CRITICAL_PERIOD_S, CRITICAL_SIZE  # noqa: E402
from plca_sim.traffic import CriticalFlow  # noqa: E402

DEADLINES_MS = (1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)
LOADS = (0.5, 0.7, 0.9)
BUDGETS = {
    "full budget": None,
    "shrink when low": (0.5, 0.5, 1.0, 1.0),
    "shrink unless congested": (0.5, 0.5, 0.75, 1.0),
}
TOPOLOGIES = ((4, 32), (8, 20), (8, 32), (8, 64), (12, 32), (16, 32))


def flows(deadline_s: float, nodes=CRITICAL_NODES) -> tuple[CriticalFlow, ...]:
    return tuple(CriticalFlow(n, CRITICAL_PERIOD_S, CRITICAL_SIZE, deadline_s) for n in nodes)


def traffic(cfg: PLCAConfig, load: float, duration: float, seed: int, deadline_s: float):
    frames = make_traffic("mixed", load, cfg, duration, seed)
    dl = cfg.s_to_bt(deadline_s)
    return [replace(f, deadline=dl) if f.critical else f for f in frames]


def deadline_job(job: tuple) -> list[dict]:
    deadline_ms, load, seed, duration = job
    cfg = PLCAConfig()
    d = deadline_ms / 1000
    frames = traffic(cfg, load, duration, seed, d)
    base = {"deadline_ms": deadline_ms, "load": load, "seed": seed}
    rows = [{**base, "policy": "conventional", **summarize(simulate(frames, cfg, cfg.s_to_bt(duration)))}]
    try:
        gov = SafetyGovernor(cfg, flows(d))
    except GovernorInfeasible:
        return rows + [{**base, "policy": "governor", "infeasible": True}]
    res = simulate(frames, cfg, cfg.s_to_bt(duration), AdaptivePolicy(cfg, gov, mode="off"))
    return rows + [{**base, "policy": "governor", "infeasible": False,
                    "bound_us": max(cfg.bt_to_us(v) for v in gov.latency_bound_bt.values()),
                    "budget_us": cfg.bt_to_us(gov.nc_budget_bt), **summarize(res)}]


def budget_job(job: tuple) -> list[dict]:
    name, load, seed, duration, model_path = job
    cfg = PLCAConfig()
    model = joblib.load(model_path)
    frames = traffic(cfg, load, duration, seed, 2e-3)
    pol = AdaptivePolicy(cfg, SafetyGovernor(cfg, flows(2e-3)), model=model, mode="tinyml",
                         budget_scale=BUDGETS[name])
    return [{"variant": name, "load": load, "seed": seed,
             **summarize(simulate(frames, cfg, cfg.s_to_bt(duration), pol))}]


def topology_job(job: tuple) -> list[dict]:
    nodes, to_bt, load, seed, duration = job
    cfg = PLCAConfig(node_count=nodes, to_timer_bt=to_bt)
    frames = traffic(cfg, load, duration, seed, 2e-3)
    base = {"nodes": nodes, "to_timer_bt": to_bt, "load": load, "seed": seed}
    conv = summarize(simulate(frames, cfg, cfg.s_to_bt(duration), ConventionalPolicy(cfg)))
    rows = [{**base, "policy": "conventional", **conv}]
    gov = SafetyGovernor(cfg, flows(2e-3))
    res = simulate(frames, cfg, cfg.s_to_bt(duration), AdaptivePolicy(cfg, gov, mode="off"))
    rows.append({**base, "policy": "governor",
                 "bound_us": max(cfg.bt_to_us(v) for v in gov.latency_bound_bt.values()), **summarize(res)})
    return rows


def run(pool, fn, jobs) -> pd.DataFrame:
    return pd.DataFrame([r for part in pool.map(fn, jobs, chunksize=1) for r in part])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--model", default="results/tinyml_model.joblib")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.seeds = 0.3, 1
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    seeds = range(300, 300 + args.seeds)  # not used anywhere else
    t0 = time.time()

    with mp.Pool(args.workers) as pool:
        print("Study 1: cost of the guarantee vs deadline")
        dl = run(pool, deadline_job, [(d, l, s, args.duration) for d in DEADLINES_MS for l in LOADS for s in seeds])
        print("Study 2: budget adaptation")
        bg = run(pool, budget_job, [(n, l, s, args.duration, args.model) for n in BUDGETS for l in LOADS for s in seeds])
        print("Study 3: topology")
        tp = run(pool, topology_job, [(n, to, l, s, args.duration) for n, to in TOPOLOGIES for l in LOADS for s in seeds])

    dl.to_csv(out / "tradeoff_deadline.csv", index=False)
    bg.to_csv(out / "tradeoff_budget.csv", index=False)
    tp.to_csv(out / "tradeoff_topology.csv", index=False)

    cols = ["noncritical_latency_mean_us", "critical_latency_max_us", "critical_deadline_miss_rate", "loss_rate"]
    s1 = dl.groupby(["load", "deadline_ms", "policy"])[cols + ["budget_us", "bound_us"]].mean(numeric_only=True)
    print("\n" + s1.round(4).to_string())
    print("\n" + bg.groupby(["load", "variant"])[cols + ["latency_p99_us"]].mean().round(4).to_string())
    print("\n" + tp.groupby(["nodes", "to_timer_bt", "load", "policy"])[cols + ["bound_us"]]
          .mean(numeric_only=True).round(4).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = {0.5: "#9ecae1", 0.7: "#4292c6", 0.9: "#084594"}
    for load in LOADS:
        g = dl[(dl.policy == "governor") & (dl.load == load) & ~dl.infeasible.fillna(False).astype(bool)]
        ms = g.groupby("deadline_ms").noncritical_latency_mean_us.mean() / 1000
        ax.plot(ms.index, ms.values, marker="o", color=colors[load], label=f"governor, {load:.0%} load")
        c = dl[(dl.policy == "conventional") & (dl.load == load)].noncritical_latency_mean_us.mean() / 1000
        ax.axhline(c, color=colors[load], ls=":", lw=1)
    ax.set_xlabel("Critical deadline (ms)")
    ax.set_ylabel("Average non-critical latency (ms)")
    ax.set_title("Cost of the guarantee (dotted: conventional PLCA)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tradeoff_deadline.png", dpi=150)
    plt.close(fig)
    print(f"\nDone in {time.time() - t0:.0f} s. Results saved in '{out}/'.")


if __name__ == "__main__":
    main()
