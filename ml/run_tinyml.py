"""Phase 4: train and evaluate the TinyML network-state predictor.

Usage (from the project folder, with the virtual environment active):
    python -m ml.run_tinyml            # full run (a few minutes)
    python -m ml.run_tinyml --quick    # short run to check everything works

Steps:
  1. Simulate conventional PLCA in all scenarios (incl. "dynamic") and record
     one feature vector per 5 ms monitoring window.
  2. Label each window with the state of the NEXT 20 ms (4 windows)
     (low / medium / high / congested).
  3. Train a small decision tree on some random seeds, test on unseen seeds.
  4. Compare with a "persistence" baseline that assumes the next window looks
     like the current one, and export the tree as C code for an ECU.

Outputs in `results/`:
    tinyml_dataset.csv          all labelled windows
    tinyml_metrics.csv          accuracy / F1 per scenario, model vs baseline
    tinyml_size_vs_accuracy.csv tree depth vs size vs accuracy
    tinyml_*.png                plots
    tinyml_model.joblib         trained model (Python)
    plca_state_model.h          trained model (C, for an ECU)
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

from plca_sim import PLCAConfig  # noqa: E402
from plca_sim.monitor import FEATURES  # noqa: E402
from plca_sim.scenarios import ALL_SCENARIOS  # noqa: E402
from plca_sim.tinyml import STATES, build_dataset, export_c, train_tree  # noqa: E402

LOADS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def scores(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--duration", type=float, default=1.0, help="simulated seconds per run")
    ap.add_argument("--train-seeds", type=int, default=4)
    ap.add_argument("--test-seeds", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=4, help="tree depth (model size)")
    ap.add_argument("--quick", action="store_true", help="0.3 s runs, 2+1 seeds")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.train_seeds, args.test_seeds = 0.3, 2, 1

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cfg = PLCAConfig(node_count=args.nodes)
    t0 = time.time()

    # 1-2. Dataset ---------------------------------------------------------
    train_seeds = tuple(range(args.train_seeds))
    test_seeds = tuple(range(100, 100 + args.test_seeds))  # never seen in training
    print("Collecting training windows...")
    train = build_dataset(ALL_SCENARIOS, LOADS, train_seeds, cfg, args.duration)
    print("Collecting test windows...")
    test = build_dataset(ALL_SCENARIOS, LOADS, test_seeds, cfg, args.duration)
    pd.concat([train.assign(split="train"), test.assign(split="test")]).to_csv(
        out / "tinyml_dataset.csv", index=False
    )
    print(f"  {len(train)} training windows, {len(test)} test windows")
    print("  label share (train):", dict(zip(STATES, np.bincount(train.state_next, minlength=4))))

    # 3. Train ---------------------------------------------------------------
    model = train_tree(train, max_depth=args.max_depth)
    test = test.copy()
    test["pred"] = model.tree.predict(test[list(FEATURES)].to_numpy())

    # 4. Evaluate ------------------------------------------------------------
    rows = []
    for sc in ("all",) + ALL_SCENARIOS:
        part = test if sc == "all" else test[test.scenario == sc]
        m = scores(part.state_next, part.pred)
        b = scores(part.state_next, part.state_now)
        rows.append({
            "scenario": sc,
            "windows": len(part),
            "tinyml_accuracy": m["accuracy"],
            "tinyml_macro_f1": m["macro_f1"],
            "persistence_accuracy": b["accuracy"],
            "persistence_macro_f1": b["macro_f1"],
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "tinyml_metrics.csv", index=False)

    # Model size vs accuracy (the TinyML trade-off).
    size_rows = []
    for depth in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        mdl = train_tree(train, max_depth=depth)
        pred = mdl.tree.predict(test[list(FEATURES)].to_numpy())
        s = scores(test.state_next, pred)
        size_rows.append({"max_depth": depth, "nodes": mdl.n_nodes, **s})
    size = pd.DataFrame(size_rows)
    size.to_csv(out / "tinyml_size_vs_accuracy.csv", index=False)

    # Export ---------------------------------------------------------------
    joblib.dump(model.tree, out / "tinyml_model.joblib")
    (out / "plca_state_model.h").write_text(export_c(model))

    # Plots ----------------------------------------------------------------
    cm = confusion_matrix(test.state_next, test.pred, labels=[0, 1, 2, 3], normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), STATES)
    ax.set_yticks(range(4), STATES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted next state")
    ax.set_ylabel("Actual next state")
    ax.set_title("TinyML confusion matrix (unseen test runs)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "tinyml_confusion_matrix.png", dpi=150)
    plt.close(fig)

    imp = pd.Series(model.tree.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Importance")
    ax.set_title("Which monitored features the model uses")
    fig.tight_layout()
    fig.savefig(out / "tinyml_feature_importance.png", dpi=150)
    plt.close(fig)

    m = metrics[metrics.scenario != "all"]
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, m.persistence_macro_f1, 0.4, label="persistence baseline")
    ax.bar(x + 0.2, m.tinyml_macro_f1, 0.4, label="TinyML decision tree")
    ax.set_xticks(x, m.scenario)
    ax.set_ylabel("Macro F1 (state of next 20 ms)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Prediction quality per scenario (unseen test runs)")
    fig.tight_layout()
    fig.savefig(out / "tinyml_per_scenario.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(size.nodes, size.macro_f1, marker="o")
    for _, r in size.iterrows():
        ax.annotate(f"d={int(r.max_depth)}", (r.nodes, r.macro_f1), textcoords="offset points",
                    xytext=(4, -10), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (tree nodes, log scale)")
    ax.set_ylabel("Macro F1")
    ax.set_title("TinyML trade-off: model size vs accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tinyml_size_vs_accuracy.png", dpi=150)
    plt.close(fig)

    # One dynamic run: actual vs predicted state over time.
    run = test[(test.scenario == "dynamic") & (test.load == 0.9) & (test.seed == test_seeds[0])]
    run = run[run.t_ms <= 500]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    a1.plot(run.t_ms, run.util, lw=1)
    a1.set_ylabel("Bus utilization")
    a1.set_title("Dynamic traffic: actual vs predicted state of the next 20 ms")
    a1.grid(alpha=0.3)
    a2.step(run.t_ms, run.state_next, where="post", label="actual", lw=1.5)
    a2.step(run.t_ms, run.pred, where="post", label="TinyML prediction", lw=1, ls="--")
    a2.set_yticks(range(4), STATES)
    a2.set_xlabel("Time (ms)")
    a2.legend(loc="upper right")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tinyml_timeline.png", dpi=150)
    plt.close(fig)

    # Summary --------------------------------------------------------------
    print("\nPrediction of the network state over the next 20 ms (unseen test runs):")
    print(metrics.round(3).to_string(index=False))
    print(f"\nModel: decision tree, depth {model.depth}, {model.n_nodes} nodes "
          f"(at most {model.depth} comparisons per prediction).")
    print(f"C code: {out / 'plca_state_model.h'}")
    print(f"\nDone in {time.time() - t0:.0f} s. Results saved in '{out}/'.")


if __name__ == "__main__":
    main()
