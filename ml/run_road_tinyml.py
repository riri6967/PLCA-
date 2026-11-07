"""Does the TinyML predictor work on real traffic?

Usage (from the project folder, with the virtual environment active):
    python -m ml.run_road_tinyml          # after run_tinyml.py
    python -m ml.run_road_tinyml --quick

Collects monitoring windows from replayed ROAD logs (conventional PLCA, same
features and labels as run_tinyml.py), then asks:
  * how well does the model trained only on synthetic traffic do on real
    traffic (domain shift)?
  * does adding ROAD windows to the training set help on logs it has never
    seen (leave-one-log-out)?

It also trains a final model on all logs except --holdout, for the closed-loop
test on those logs: python -m traffic.run_road --model results/tinyml_model_road.joblib
--log <held-out logs> --out results/road_heldout

Outputs in `results/`: road_tinyml_metrics.csv, road_tinyml_per_log.csv,
road_tinyml_windows.csv, tinyml_model_road.joblib, plca_state_model_road.h
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from plca_sim import PLCAConfig
from plca_sim.monitor import FEATURES
from plca_sim.road import make_road_traffic, read_candump
from plca_sim.tinyml import export_c, monitor_windows, train_tree

AMBIENT = Path("dataset") / "road" / "road" / "ambient"
HOLDOUT = ("highway_street_driving_long", "dyno_drive_winter", "dyno_idle_radio_infotainment")
# (CAN segments, bulk Ethernet load)
MIXES = ((1, 0.0), (2, 0.0), (4, 0.0), (6, 0.0), (8, 0.0), (1, 0.3), (2, 0.3), (3, 0.3), (4, 0.3))


def windows_of_log(job: tuple) -> pd.DataFrame:
    path, runs, duration, max_trace = job
    name = Path(path).stem.replace("ambient_", "")
    cfg = PLCAConfig()
    trace = read_candump(path, max_trace)
    if trace.t.max() <= duration:
        return pd.DataFrame()
    parts = []
    for copies, bulk in MIXES:
        for seed in range(runs):
            rt = make_road_traffic(trace, cfg, copies, duration, seed, bulk_load=bulk)
            df = monitor_windows(rt.frames, cfg, duration)
            if not df.empty:
                parts.append(df.assign(log=name, copies=copies, bulk=bulk, seed=seed))
    print(f"  {name}: {sum(len(p) for p in parts)} windows", flush=True)
    return pd.concat(parts, ignore_index=True)


def scores(y_true, y_pred) -> dict[str, float]:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--runs", type=int, default=2, help="replays per load level")
    ap.add_argument("--max-trace", type=float, default=300.0)
    ap.add_argument("--model", default="results/tinyml_model.joblib")
    ap.add_argument("--synthetic", default="results/tinyml_dataset.csv")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.runs, args.max_trace = 0.3, 1, 60.0
    out = Path(args.out)
    t0 = time.time()

    logs = sorted(AMBIENT.glob("*.log"))
    print(f"Collecting windows from {len(logs)} logs...")
    with mp.Pool(args.workers) as pool:
        parts = pool.map(windows_of_log, [(str(p), args.runs, args.duration, args.max_trace) for p in logs], chunksize=1)
    road = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    road.to_csv(out / "road_tinyml_windows.csv", index=False)
    print(f"  {len(road)} windows; states: {road.state_next.value_counts().sort_index().to_dict()}")

    synth = pd.read_csv(args.synthetic)
    synth_train = synth[synth.split == "train"]
    synth_test = synth[synth.split == "test"]
    X = list(FEATURES)
    base = joblib.load(args.model)

    # Leave one log out: every prediction is on a log the model never saw.
    preds = {"synthetic model": [], "road only": [], "synthetic + road": [], "persistence": []}
    truth, log_of = [], []
    for name in sorted(road.log.unique()):
        test = road[road.log == name]
        train = road[road.log != name]
        truth.append(test.state_next.to_numpy())
        log_of += [name] * len(test)
        preds["synthetic model"].append(base.predict(test[X].to_numpy()))
        preds["road only"].append(train_tree(train).tree.predict(test[X].to_numpy()))
        both = pd.concat([synth_train, train], ignore_index=True)
        preds["synthetic + road"].append(train_tree(both).tree.predict(test[X].to_numpy()))
        preds["persistence"].append(test.state_now.to_numpy())
    y = np.concatenate(truth)
    rows, per_log = [], []
    logs_arr = np.array(log_of)
    for name, chunks in preds.items():
        p = np.concatenate(chunks)
        rows.append({"model": name, "tested_on": "ROAD (unseen log)", "windows": len(y), **scores(y, p),
                     "within_one_class": float(np.mean(np.abs(y - p) <= 1))})
        for log in sorted(set(log_of)):
            sel = logs_arr == log
            per_log.append({"model": name, "log": log, "windows": int(sel.sum()), **scores(y[sel], p[sel])})

    # Final models, trained without the held-out logs (for the closed-loop test).
    train = road[~road.log.isin(HOLDOUT)]
    final = train_tree(pd.concat([synth_train, train], ignore_index=True))
    rows.append({"model": "synthetic + road", "tested_on": "synthetic test seeds", "windows": len(synth_test),
                 **scores(synth_test.state_next, final.tree.predict(synth_test[X].to_numpy()))})
    rows.append({"model": "synthetic model", "tested_on": "synthetic test seeds", "windows": len(synth_test),
                 **scores(synth_test.state_next, base.predict(synth_test[X].to_numpy()))})
    joblib.dump(final.tree, out / "tinyml_model_road.joblib")
    (out / "plca_state_model_road.h").write_text(export_c(final), encoding="utf-8")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "road_tinyml_metrics.csv", index=False)
    pd.DataFrame(per_log).to_csv(out / "road_tinyml_per_log.csv", index=False)
    print("\n" + metrics.round(3).to_string(index=False))
    print(f"\nFinal model: {final.n_nodes} nodes, depth {final.depth}, trained without {HOLDOUT}")
    print(f"Done in {time.time() - t0:.0f} s. Results saved in '{out}/'.")


if __name__ == "__main__":
    main()
