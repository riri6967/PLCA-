"""Reduce one run's result files to one CSV row of plan metrics.

    python3 sim/collect.py results/run.sca [--deadline-ms 2] [--safety 0,1] [--keep]

The .vec file holds one latency per frame and grows with run length, so it is
removed afterwards unless --keep is given. Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics as st
from pathlib import Path

FIELDS = ["run", "config", "load", "maxBc", "numNodes", "seed", "frames", "lat_mean_ms", "lat_p99_ms",
          "lat_max_ms", "jitter_ms", "throughput_mbps", "fairness_jain", "drops", "drop_rate",
          "bus_utilization", "safety_frames", "safety_drops", "safety_miss_rate", "safety_miss_2ms", "safety_miss_5ms",
          "safety_miss_10ms", "safety_max_ms", "cycle_mean_ms", "cycle_max_ms", "to_utilization", "queue_mean",
          "queue_max", "proposals", "overrides", "override_rate", "changes", "safety_miss_20ms"]
# Bytes each delivered frame adds to its payload: Ethernet header and FCS, preamble, gap.
FRAME_OVERHEAD = 18 + 8 + 12


def percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def read_vec(path: Path) -> dict[int, list[float]]:
    """Latency values per follower, from the sink named controller.app[i]."""
    ids: dict[str, int] = {}
    out: dict[int, list[float]] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("vector "):
                parts = line.split()
                m = re.search(r"controller\.app\[(\d+)\]", parts[2])
                if m and "meanBitLifeTimePerPacket" in parts[3]:
                    ids[parts[1]] = int(m.group(1))
                    out.setdefault(int(m.group(1)), [])
            elif line[:1].isdigit():
                parts = line.split()
                if parts[0] in ids and len(parts) >= 4:
                    out[ids[parts[0]]].append(float(parts[3]))
    return out


def read_sca(path: Path) -> tuple[dict[str, str], dict[str, float]]:
    attrs: dict[str, str] = {}
    scalars: dict[str, float] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith(("attr ", "itervar ", "config ")):
                parts = line.rstrip("\n").split(" ", 2)
                if len(parts) == 3:
                    attrs[parts[1]] = parts[2].strip('"')
            elif line.startswith("scalar "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        scalars[parts[1] + " " + parts[2]] = float(parts[-1])
                    except ValueError:
                        pass
    return attrs, scalars


def total(scalars: dict[str, float], module: str, name: str) -> float:
    return sum(v for k, v in scalars.items() if re.search(module, k.split()[0]) and k.split()[1] == name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sca", type=Path)
    ap.add_argument("--deadline-ms", type=float, default=2.0, help="deadline of safety_miss_rate")
    ap.add_argument("--safety", default="0,1", help="follower indices with safety flows; empty for none")
    ap.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    attrs, sc = read_sca(args.sca)
    vec = args.sca.with_suffix(".vec")
    lat = read_vec(vec)
    all_ms = sorted(v * 1e3 for vals in lat.values() for v in vals)
    safety = [int(x) for x in args.safety.split(",") if x != ""]
    safe_ms = [v * 1e3 for n in safety for v in lat.get(n, [])]
    per_node_std = [st.pstdev(v) * 1e3 for v in lat.values() if len(v) > 1]

    sent: dict[int, float] = {}
    for k, v in sc.items():
        mod, name = k.split()
        m = re.search(r"\.node\[(\d+)\]\.app\[\d+\]", mod)
        if m and name == "packetSent:count":
            sent[int(m.group(1))] = sent.get(int(m.group(1)), 0.0) + v
    served = [len(lat.get(n, [])) / s for n, s in sent.items() if s]
    jain = (sum(served) ** 2 / (len(served) * sum(x * x for x in served))) if served and any(served) else math.nan

    drops = total(sc, r"\.node\[\d+\]\.eth\[0\]\.queue", "droppedPacketsQueueOverflow:count")
    frames = len(all_ms)
    sim_s = float(attrs.get("sim-time-limit", "60s").rstrip("s"))
    warm_s = float(attrs.get("warmup-period", "5s").rstrip("s"))
    received = total(sc, r"\.controller\.app\[\d+\]", "packetReceived:sum(packetBytes)")
    span = max(sim_s - warm_s, 1e-9)
    plca = r"\.controller\.eth\[0\]\.plca"
    queues = r"\.node\[\d+\]\.eth\[0\]\.queue"
    means = [v for k, v in sc.items() if re.search(queues, k.split()[0]) and k.split()[1] == "queueLength:timeavg"]
    highs = [v for k, v in sc.items() if re.search(queues, k.split()[0]) and k.split()[1] == "queueLength:max"]
    safe_drops = sum(total(sc, rf"\.node\[{n}\]\.eth\[0\]\.queue", "droppedPacketsQueueOverflow:count") for n in safety)

    # Every node reports only its own TO, the coordinator's are always empty.
    used = [v for k, v in sc.items() if re.search(r"\.(node\[\d+\]|controller)\.eth\[0\]\.plca", k.split()[0])
            and k.split()[1] == "transmitOpportunityUsed:mean"]

    def missed(deadline_ms: float) -> float:
        """A dropped frame is a miss too: leaving it out hides the worst failures."""
        known = len(safe_ms) + safe_drops
        return (sum(1 for v in safe_ms if v > deadline_ms) + safe_drops) / known if known else math.nan

    def controller(name: str) -> float:
        hits = [v for k, v in sc.items() if ".controller" in k and k.split()[1] == name]
        return hits[0] if hits else math.nan

    row = {
        "run": args.sca.stem, "config": attrs.get("configname", ""), "load": attrs.get("load", ""),
        "maxBc": attrs.get("maxBc", ""), "numNodes": attrs.get("nodes") or attrs.get("numNodes", ""),
        "seed": attrs.get("repetition", ""), "frames": frames,
        "lat_mean_ms": sum(all_ms) / frames if frames else math.nan,
        "lat_p99_ms": percentile(all_ms, 0.99) if frames else math.nan,
        "lat_max_ms": all_ms[-1] if frames else math.nan,
        "jitter_ms": sum(per_node_std) / len(per_node_std) if per_node_std else math.nan,
        "throughput_mbps": received * 8 / span / 1e6, "fairness_jain": jain, "drops": drops,
        "drop_rate": drops / (frames + drops) if frames + drops else 0.0,
        "bus_utilization": (received + FRAME_OVERHEAD * frames) * 8 / span / 1e7,
        "safety_frames": len(safe_ms), "safety_drops": safe_drops, "safety_miss_rate": missed(args.deadline_ms),
        "safety_miss_2ms": missed(2), "safety_miss_5ms": missed(5), "safety_miss_10ms": missed(10),
        "safety_max_ms": max(safe_ms) if safe_ms else math.nan,
        "cycle_mean_ms": total(sc, plca, "cycleLength:mean") * 1e3,
        "cycle_max_ms": total(sc, plca, "cycleLength:max") * 1e3,
        "to_utilization": st.fmean(used) if used else math.nan,
        "queue_mean": sum(means) / len(means) if means else math.nan, "queue_max": max(highs) if highs else math.nan,
        "proposals": controller("proposals"), "overrides": controller("overrides"),
        "override_rate": controller("overrideRate"), "changes": controller("changes"),
        "safety_miss_20ms": missed(20),
    }

    new = not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    if not args.keep and vec.exists():
        vec.unlink()
        vci = vec.with_suffix(".vci")
        if vci.exists():
            vci.unlink()
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
