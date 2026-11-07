"""Average results/summary.csv over seeds, with a 95% confidence interval.

    python3 summarize.py results/summary.csv --out results/summary_ci.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

METRICS = ("lat_mean_ms", "lat_p99_ms", "lat_max_ms", "jitter_ms", "throughput_mbps", "fairness_jain",
           "drop_rate", "safety_miss_rate", "safety_max_ms")
KEYS = ("config", "load", "maxBc", "numNodes")
# Two-sided 97.5% Student t values by degrees of freedom. Beyond 30, 2.0 is close enough.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 19: 2.093, 29: 2.042}


def t_value(df: int) -> float:
    return T975.get(df, 2.0 if df > 29 else T975[min(k for k in T975 if k >= df)])


def mean_ci(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return math.nan, math.nan
    if len(clean) == 1:
        return clean[0], math.nan
    half = t_value(len(clean) - 1) * statistics.stdev(clean) / math.sqrt(len(clean))
    return statistics.fmean(clean), half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("summary")
    ap.add_argument("--out")
    args = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(args.summary, newline="") as fh:
        for row in csv.DictReader(fh):
            groups[tuple(row[k] for k in KEYS)].append(row)

    rows = []
    for key in sorted(groups, key=lambda k: (k[0], float(k[1] or 0), int(k[2] or 0), int(k[3] or 0))):
        out = dict(zip(KEYS, key), seeds=len(groups[key]))
        for m in METRICS:
            mean, half = mean_ci([float(r[m]) for r in groups[key]])
            out[m] = mean
            out[m + "_ci95"] = half
        out["worst_lat_max_ms"] = max(float(r["lat_max_ms"]) for r in groups[key])
        rows.append(out)

    print(f"{'config':16}{'load':>6}{'bc':>3}{'n':>3}{'seeds':>6}{'mean ms':>16}{'p99 ms':>16}"
          f"{'max ms':>9}{'Mbit/s':>8}{'drop':>8}{'Jain':>7}{'miss':>8}{'safety max':>11}")
    for r in rows:
        print(f"{r['config']:16}{r['load']:>6}{r['maxBc']:>3}{r['numNodes']:>3}{r['seeds']:>6}"
              f"{r['lat_mean_ms']:>9.3f}±{r['lat_mean_ms_ci95']:<6.3f}{r['lat_p99_ms']:>9.3f}±{r['lat_p99_ms_ci95']:<6.3f}"
              f"{r['worst_lat_max_ms']:>9.2f}{r['throughput_mbps']:>8.3f}{r['drop_rate']:>8.4f}{r['fairness_jain']:>7.4f}"
              f"{r['safety_miss_rate']:>8.4f}{r['safety_max_ms']:>11.2f}")

    if args.out and rows:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
