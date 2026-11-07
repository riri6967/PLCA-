"""Work Plan Phase 5 in the Python simulator: baselines B1-B4, proposed, ablation.

These are interim results: the plan's final numbers must come from OMNeT++/INET.

Usage (after run_tinyml.py):
    python -m analysis.run_plan_eval          # 10 seeds of 20 s, first 2 s discarded
    python -m analysis.run_plan_eval --quick

Safety flows: 2 nodes, 64 bytes every 1 ms, deadline 10 ms. The standards-only
governor cannot guarantee less on this segment (run_plan_eval prints why).

Outputs in `results/`: plan_results.csv, plan_summary.csv (mean, 95 % CI)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from plca_sim import PLCAConfig, make_traffic, simulate, summarize
from plca_sim.plan import PlanGovernor, PlanPolicy, PlcaSettings, safest_settings
from plca_sim.policy import CyclePlan, Policy
from plca_sim.scenarios import CRITICAL_NODES, CRITICAL_PERIOD_S, CRITICAL_SIZE
from plca_sim.traffic import CriticalFlow

DEADLINE_S = 10e-3
RUNS = [("S1 low", "uniform", 0.25)]
RUNS += [("S2 increasing", "uniform", x) for x in (0.3, 0.5, 0.7, 0.9)]
RUNS += [("S3 bursty", "bursty", x) for x in (0.3, 0.5, 0.7, 0.9)]
RUNS += [("S4 mixed", "mixed", x) for x in (0.3, 0.5, 0.7, 0.9)]
STATIC_BURSTS = (0, 1, 3, 7)
SAFETY_LAST = (0, 3, 4, 5, 6, 7, 1, 2)


class Fixed(Policy):
    """Static PLCA settings."""

    def __init__(self, cfg, s: PlcaSettings):
        super().__init__(cfg)
        self.s = s

    def plan(self, obs) -> CyclePlan:
        return CyclePlan([1 + b for b in self.s.burst], order=self.s.order)

    def frames_per_to(self, obs):
        return self.plan(obs).limits


def wrong_state(rng):
    return lambda feat: int(rng.integers(0, 4))


def governor(cfg) -> PlanGovernor:
    flows = [CriticalFlow(n, CRITICAL_PERIOD_S, CRITICAL_SIZE, DEADLINE_S) for n in CRITICAL_NODES]
    frames = [1518] * cfg.node_count
    for n in CRITICAL_NODES:
        frames[n] = CRITICAL_SIZE
    return PlanGovernor(cfg, flows, frames)


def policies(cfg, scenario: str, model, static_b: int, seed: int) -> dict[str, Policy]:
    gov = governor(cfg)
    default = PlcaSettings.default(cfg.node_count)
    safe = safest_settings(gov) if scenario == "mixed" else default
    guard = gov if scenario == "mixed" else None
    tuned = tuple((b if n not in CRITICAL_NODES or scenario != "mixed" else max(b, s))
                  for n, (b, s) in enumerate(zip([static_b] * cfg.node_count, safe.burst)))
    rng = np.random.default_rng(seed)
    last = PlcaSettings(safe.burst, SAFETY_LAST)
    return {
        "B1 conventional": Fixed(cfg, default),
        "B2 best static": Fixed(cfg, PlcaSettings(tuned, default.order)),
        "B3 fixed priority": Fixed(cfg, safe),
        "B4 reactive EWMA": PlanPolicy(cfg, safe, guard, "ewma"),
        "P proposed": PlanPolicy(cfg, safe, guard, "tinyml", model),
        "A1 no governor": PlanPolicy(cfg, safe, None, "tinyml", model),
        "A2 no reordering": PlanPolicy(cfg, last, guard, "tinyml", model),
        "A3 wrong predictions": PlanPolicy(cfg, safe, guard, wrong_state(rng)),
    }


def traffic(scenario, load, cfg, duration, seed):
    frames = make_traffic(scenario, load, cfg, duration, seed)
    dl = cfg.s_to_bt(DEADLINE_S)
    return [replace(f, deadline=dl) if f.critical else f for f in frames]


def best_static(job) -> tuple:
    """Offline tuning (B2): lowest p99 over tuning seeds never used for testing."""
    scenario, load, duration = job
    cfg = PLCAConfig()
    score = {}
    for b in STATIC_BURSTS:
        vals = []
        for seed in (900, 901):
            frames = traffic(scenario, load, cfg, duration, seed)
            pol = policies(cfg, scenario, None, b, seed)["B2 best static"]
            vals.append(summarize(simulate(frames, cfg, cfg.s_to_bt(duration), pol))["latency_p99_us"])
        score[b] = float(np.mean(vals))
    return (scenario, load), min(score, key=score.get)


def one(job) -> list[dict]:
    label, scenario, load, seed, duration, warmup, static_b, model_path = job
    cfg = PLCAConfig()
    model = joblib.load(model_path)
    frames = traffic(scenario, load, cfg, duration, seed)
    rows = []
    for name, pol in policies(cfg, scenario, model, static_b, seed).items():
        res = simulate(frames, cfg, cfg.s_to_bt(duration), pol)
        m = summarize(res, cfg.s_to_bt(warmup))
        m.update({"policy": name, "scenario": label, "load": load, "seed": seed, "static_burst": static_b,
                  "override_rate": getattr(pol, "override_rate", float("nan")),
                  "reconfigurations": getattr(pol, "changes", 0)})
        rows.append(m)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--model", default="results/tinyml_model.joblib")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.warmup, args.seeds = 0.5, 0.05, 2
    t0 = time.time()

    print("Safety bound on this segment: ", end="")
    gov = governor(PLCAConfig())
    s = safest_settings(gov)
    print(f"safety burst {s.burst[1]}, latency bound {max(gov.latency_bound_bt(s).values()) / 1e4:.2f} ms, "
          f"deadline {DEADLINE_S * 1e3:.0f} ms", flush=True)

    with mp.Pool(args.workers) as pool:
        tuned = dict(pool.map(best_static, sorted({(sc, l, min(args.duration, 5.0)) for _, sc, l in RUNS}), chunksize=1))
        print("B2 tuning done", tuned, flush=True)
        jobs = [(lab, sc, l, 500 + i, args.duration, args.warmup, tuned[(sc, l)], args.model)
                for lab, sc, l in RUNS for i in range(args.seeds)]
        rows = [r for part in pool.imap_unordered(one, jobs, chunksize=1) for r in part]

    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out / "plan_results.csv", index=False)
    cols = ["latency_mean_us", "latency_p99_us", "jitter_us", "throughput_mbps", "fairness_jain", "loss_rate",
            "critical_deadline_miss_rate", "critical_latency_max_us", "override_rate", "reconfigurations"]
    g = df.groupby(["scenario", "load", "policy"])[cols]
    summary = g.mean()
    ci = 1.96 * g.std() / np.sqrt(args.seeds)  # normal approximation
    table = summary.join(ci.add_suffix("_ci95")).reset_index()
    table["critical_latency_worst_us"] = df.groupby(["scenario", "load", "policy"]).critical_latency_max_us.max().to_numpy()
    table.to_csv(out / "plan_summary.csv", index=False)
    show = ["scenario", "load", "policy", "latency_p99_us", "latency_p99_us_ci95", "fairness_jain",
            "critical_deadline_miss_rate", "critical_latency_worst_us", "override_rate"]
    print(table[show].round(3).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
