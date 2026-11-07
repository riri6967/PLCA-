"""How much better could any predictor be? Run after run_tinyml.py.

Compares the TinyML tree with two references on the same unseen test windows:
  * gradient boosting: a much larger model
  * oracle: knows the true scenario and load of the run, and always answers
    the state most common for that scenario and load in the training data

Output: results/tinyml_reference_models.csv
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from plca_sim.monitor import FEATURES


def row(name: str, y, p) -> dict:
    return {"model": name, "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0),
            "within_one_class": float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) <= 1))}


def main() -> None:
    df = pd.read_csv("results/tinyml_dataset.csv")
    train, test = df[df.split == "train"], df[df.split == "test"]
    X, y = list(FEATURES), test.state_next.to_numpy()

    tree = joblib.load("results/tinyml_model.joblib")
    boost = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0)
    boost.fit(train[X].to_numpy(), train.state_next.to_numpy())

    best = train.groupby(["scenario", "load"]).state_next.agg(lambda s: s.mode().iloc[0]).rename("oracle")
    oracle = test.join(best, on=["scenario", "load"]).oracle.to_numpy()

    out = pd.DataFrame([
        row("TinyML tree", y, tree.predict(test[X].to_numpy())),
        row("persistence", y, test.state_now.to_numpy()),
        row("gradient boosting", y, boost.predict(test[X].to_numpy())),
        row("oracle (true scenario and load)", y, oracle),
    ])
    out.to_csv("results/tinyml_reference_models.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
