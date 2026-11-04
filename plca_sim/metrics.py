"""Turn a SimResult into the KPIs listed in the research plan."""

from __future__ import annotations

import numpy as np

from .simulator import SimResult


def summarize(res: SimResult) -> dict[str, float]:
    cfg = res.cfg
    us = cfg.bt_to_us
    recs = res.records
    duration_s = res.duration_bt / cfg.bit_rate

    lat = np.array([r.end - r.arrival for r in recs]) if recs else np.zeros(1)
    lat_us = us(lat)

    # Jitter: standard deviation of latency, averaged over nodes.
    per_node_std = []
    for n in range(cfg.node_count):
        ln = [r.end - r.arrival for r in recs if r.node == n]
        if len(ln) > 1:
            per_node_std.append(float(np.std(us(np.array(ln)))))

    crit = [r for r in recs if r.critical]
    crit_lat_us = us(np.array([r.end - r.arrival for r in crit])) if crit else np.zeros(0)
    crit_miss = [r for r in crit if r.deadline is not None and r.end - r.arrival > r.deadline]
    noncrit = [r for r in recs if not r.critical]
    noncrit_lat_us = (
        us(np.array([r.end - r.arrival for r in noncrit])) if noncrit else np.zeros(0)
    )

    delivered_bits = sum(r.size * 8 for r in recs if r.end <= res.duration_bt)
    q_avg = [s / c for s, c in zip(res.queue_sum, res.queue_samples) if c]

    return {
        "frames_offered": res.offered,
        "frames_delivered": len(recs),
        "frames_dropped": len(res.dropped),
        "frames_backlog": res.backlog,
        "loss_rate": len(res.dropped) / res.offered if res.offered else 0.0,
        "latency_mean_us": float(np.mean(lat_us)),
        "latency_p50_us": float(np.percentile(lat_us, 50)),
        "latency_p99_us": float(np.percentile(lat_us, 99)),
        "latency_max_us": float(np.max(lat_us)),
        "jitter_us": float(np.mean(per_node_std)) if per_node_std else 0.0,
        "throughput_mbps": delivered_bits / duration_s / 1e6,
        "bus_utilization": res.busy_bt / res.duration_bt,
        "to_utilization": res.tos_used / res.tos_total if res.tos_total else 0.0,
        "queue_mean": float(np.mean(q_avg)) if q_avg else 0.0,
        "cycle_mean_us": float(us(np.mean(res.cycle_lengths))) if res.cycle_lengths else 0.0,
        "cycle_max_us": float(us(np.max(res.cycle_lengths))) if res.cycle_lengths else 0.0,
        "critical_frames": len(crit),
        "critical_latency_mean_us": float(np.mean(crit_lat_us)) if len(crit_lat_us) else np.nan,
        "critical_latency_max_us": float(np.max(crit_lat_us)) if len(crit_lat_us) else np.nan,
        "critical_deadline_miss_rate": len(crit_miss) / len(crit) if crit else np.nan,
        "noncritical_latency_mean_us": (
            float(np.mean(noncrit_lat_us)) if len(noncrit_lat_us) else np.nan
        ),
    }
