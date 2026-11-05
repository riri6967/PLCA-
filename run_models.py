"""Work Plan 3.2: compare predictors, including one that uses no ML.

Run after run_tinyml.py. On the same unseen windows it compares a decision
tree, a random forest, a tiny MLP (2 layers of 16 neurons, also quantized to
int8) and a threshold rule on the smoothed utilization (EWMA, no ML).

Output: results/model_comparison.csv
"""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from plca_sim.monitor import FEATURES
from plca_sim.tinyml import STATES, state_of


def quantized_predict(mlp: MLPClassifier, x: np.ndarray) -> np.ndarray:
    """Forward pass with weights rounded to int8, as an MCU would store them."""
    a = x
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        scale = max(np.abs(w).max(), np.abs(b).max()) / 127
        a = a @ (np.round(w / scale) * scale) + np.round(b / scale) * scale
        if i < len(mlp.coefs_) - 1:
            a = np.maximum(a, 0)
    return mlp.classes_[np.argmax(a, axis=1)]


def row(name, y, pred, size_bytes, seconds=None) -> dict:
    per_class = f1_score(y, pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    out = {"model": name, "accuracy": accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0),
           "size_bytes": size_bytes, "python_us_per_window": seconds}
    out.update({f"f1_{s}": v for s, v in zip(STATES, per_class)})
    return out


def timed(fn, x) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    pred = fn(x)
    return pred, (time.perf_counter() - t0) / len(x) * 1e6


def main() -> None:
    df = pd.read_csv("results/tinyml_dataset.csv")
    train, test = df[df.split == "train"], df[df.split == "test"]
    cols = list(FEATURES)
    x_tr, x_te = train[cols].to_numpy(), test[cols].to_numpy()
    y_tr, y = train.state_next.to_numpy(), test.state_next.to_numpy()

    rows = []
    tree = joblib.load("results/tinyml_model.joblib")
    pred, us = timed(tree.predict, x_te)
    # Each node: feature index and threshold, or class. About 8 bytes each.
    rows.append(row("decision tree, depth 4", y, pred, tree.tree_.node_count * 8, us))

    forest = RandomForestClassifier(n_estimators=20, max_depth=6, class_weight="balanced",
                                    random_state=0, n_jobs=4).fit(x_tr, y_tr)
    pred, us = timed(forest.predict, x_te)
    rows.append(row("random forest, 20 trees", y, pred, sum(t.tree_.node_count for t in forest.estimators_) * 8, us))

    scale = StandardScaler().fit(x_tr)
    mlp = MLPClassifier((16, 16), max_iter=300, random_state=0).fit(scale.transform(x_tr), y_tr)
    weights = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    z = scale.transform(x_te)
    pred, us = timed(mlp.predict, z)
    rows.append(row("MLP 2x16, float32", y, pred, weights * 4, us))
    pred, us = timed(lambda v: quantized_predict(mlp, v), z)
    rows.append(row("MLP 2x16, int8", y, pred, weights, us))

    rows.append(row("EWMA threshold, no ML", y, state_of(test.util_ewma, test.queue_total), 0))
    rows.append(row("persistence, no ML", y, test.state_now.to_numpy(), 0))

    out = pd.DataFrame(rows)
    out.to_csv("results/model_comparison.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
