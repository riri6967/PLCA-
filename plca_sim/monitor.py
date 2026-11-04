"""PLCA monitoring and feature extraction (framework blocks 1 and 2).

`NetworkMonitor` turns the per-cycle `Observation`s into one feature vector
per time window. Every feature uses only counters a real PLCA node or the
coordinator could keep cheaply: queue lengths, used/total TOs, bus-busy
time, frame arrivals and the cycle length.

`MonitoringPolicy` wraps any policy and records features while the
simulation runs, without changing how the bus behaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import PLCAConfig
from .policy import ConventionalPolicy, Observation, Policy

FEATURES = (
    "util",  # bus utilization in the last window
    "util_prev",  # bus utilization in the window before
    "util_trend",  # util - util_prev
    "to_util",  # used TOs / total TOs in the last window
    "arrival_rate",  # frames arriving per ms in the last window
    "arrival_trend",  # change of arrival_rate vs. previous window
    "queue_total",  # frames waiting on all nodes now
    "queue_max",  # longest single queue now
    "active_nodes",  # nodes with a non-empty queue now
    "critical_waiting",  # critical frames waiting now
    "cycle_us",  # length of the last PLCA cycle
    "cycle_ewma_us",  # smoothed cycle length
    "util_ewma",  # smoothed bus utilization (~last 20 ms)
    "arrival_ewma",  # smoothed arrival rate
    "queue_ewma",  # smoothed total queue length
)


@dataclass
class NetworkMonitor:
    cfg: PLCAConfig
    window_s: float = 5e-3
    ewma_alpha: float = 0.2  # cycle-length smoothing (per cycle)
    window_alpha: float = 0.3  # smoothing of window features (per window)
    samples: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._window_bt = self.cfg.s_to_bt(self.window_s)
        self._last: Observation | None = None
        self._util_prev = 0.0
        self._rate_prev = 0.0
        self._cycle_ewma = float(self.cfg.empty_cycle_bt)
        self._util_ewma: float | None = None
        self._rate_ewma = 0.0
        self._queue_ewma = 0.0
        self.latest: dict[str, float] | None = None

    def observe(self, obs: Observation) -> dict[str, float] | None:
        """Feed one cycle; returns a new feature vector when a window closes."""
        self._cycle_ewma += self.ewma_alpha * (obs.last_cycle_bt - self._cycle_ewma)
        if self._last is None:
            self._last = obs
            return None
        dt = obs.now_bt - self._last.now_bt
        if dt < self._window_bt:
            return None

        last = self._last
        util = min((obs.busy_bt - last.busy_bt) / dt, 1.0)
        d_tos = obs.tos_total - last.tos_total
        to_util = (obs.tos_used - last.tos_used) / d_tos if d_tos else 0.0
        rate = (obs.arrived - last.arrived) / self.cfg.bt_to_us(dt) * 1000.0
        q = obs.queue_lengths
        a = self.window_alpha
        if self._util_ewma is None:
            self._util_ewma, self._rate_ewma, self._queue_ewma = util, rate, float(sum(q))
        else:
            self._util_ewma += a * (util - self._util_ewma)
            self._rate_ewma += a * (rate - self._rate_ewma)
            self._queue_ewma += a * (sum(q) - self._queue_ewma)
        feat = {
            "t_ms": self.cfg.bt_to_us(obs.now_bt) / 1000.0,
            "window_ms": self.cfg.bt_to_us(dt) / 1000.0,
            "util": util,
            "util_prev": self._util_prev,
            "util_trend": util - self._util_prev,
            "to_util": to_util,
            "arrival_rate": rate,
            "arrival_trend": rate - self._rate_prev,
            "queue_total": float(sum(q)),
            "queue_max": float(max(q)),
            "active_nodes": float(sum(1 for x in q if x)),
            "critical_waiting": float(sum(obs.critical_waiting)),
            "cycle_us": self.cfg.bt_to_us(obs.last_cycle_bt),
            "cycle_ewma_us": self.cfg.bt_to_us(self._cycle_ewma),
            "util_ewma": self._util_ewma,
            "arrival_ewma": self._rate_ewma,
            "queue_ewma": self._queue_ewma,
        }
        self._util_prev = util
        self._rate_prev = rate
        self._last = obs
        self.samples.append(feat)
        self.latest = feat
        return feat


class MonitoringPolicy(Policy):
    """Records monitor features while delegating decisions to `inner`."""

    def __init__(self, cfg: PLCAConfig, inner: Policy | None = None, window_s: float = 5e-3):
        super().__init__(cfg)
        self.inner = inner or ConventionalPolicy(cfg)
        self.monitor = NetworkMonitor(cfg, window_s)

    def frames_per_to(self, obs: Observation) -> list[int]:
        self.monitor.observe(obs)
        return self.inner.frames_per_to(obs)
