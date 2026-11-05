"""Work-plan design: adapt only standard PLCA parameters, behind a safety check.

Unlike `adaptive.SafetyGovernor`, which meters bus time inside every node, this
design (Work Plan, Phase 4) changes only values a stock IEEE 802.3cg PHY has:
the burst count of each node and the order of the node IDs. Before a
configuration is applied, the governor computes its worst case and rejects it
if a safety deadline could break. It then keeps the last safe configuration
and logs an override.

Worst-case cycle, with b_i the burst count of node i and F_i the largest frame
of node i (bus time, including the inter-packet gap):

    C = T_beacon + sum_i (1 + b_i) F_i

A safety node s with period P sends 1 + b_s frames per TO. Its queue is stable
only if ceil(C / P) <= 1 + b_s; a frame is then delivered within

    L = C + (1 + b_s) F_s  <= D

The Work Plan's check is L = C + F_max <= D. It leaves out the queue condition,
without which a frame can miss its deadline however long D is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import PLCAConfig
from .monitor import FEATURES, NetworkMonitor
from .policy import CyclePlan, Observation, Policy
from .tinyml import state_of
from .traffic import CriticalFlow


@dataclass(frozen=True)
class PlcaSettings:
    """Values that exist as PLCA parameters on a stock PHY."""

    burst: tuple[int, ...]
    order: tuple[int, ...]  # node IDs in TO order; the coordinator stays first

    @classmethod
    def default(cls, n: int) -> "PlcaSettings":
        return cls((0,) * n, tuple(range(n)))


class PlanGovernor:
    """Accepts a configuration only if its worst case meets every deadline."""

    def __init__(self, cfg: PLCAConfig, flows: Sequence[CriticalFlow], max_frame_bytes: Sequence[int]):
        if len(max_frame_bytes) != cfg.node_count:
            raise ValueError("need one maximum frame size per node")
        self.cfg = cfg
        self.flows = tuple(flows)
        self.frame_bt = [cfg.wire_bits(b) for b in max_frame_bytes]

    def worst_cycle_bt(self, s: PlcaSettings) -> float:
        return self.cfg.beacon_bt + sum((1 + b) * f for b, f in zip(s.burst, self.frame_bt))

    def latency_bound_bt(self, s: PlcaSettings) -> dict[int, float]:
        """Latency bound per safety node; infinite if its queue can grow forever."""
        cycle = self.worst_cycle_bt(s)
        bounds: dict[int, float] = {}
        per_node: dict[int, int] = {}
        for f in self.flows:
            per_node[f.node] = per_node.get(f.node, 0) + math.ceil(cycle / self.cfg.s_to_bt(f.period_s))
        for f in self.flows:
            room = 1 + s.burst[f.node]
            stable = per_node[f.node] <= room
            bounds[f.node] = cycle + room * self.cfg.wire_bits(f.size) if stable else math.inf
        return bounds

    def accepts(self, s: PlcaSettings) -> bool:
        bounds = self.latency_bound_bt(s)
        return all(bounds[f.node] <= self.cfg.s_to_bt(f.deadline_s) for f in self.flows)


def safest_settings(gov: PlanGovernor, order: tuple[int, ...] | None = None) -> PlcaSettings | None:
    """Smallest safety-node bursts the governor accepts, others at 0, if any."""
    cfg = gov.cfg
    order = order or tuple(range(cfg.node_count))
    nodes = sorted({f.node for f in gov.flows})
    for extra in range(64):
        burst = [extra if n in nodes else 0 for n in range(cfg.node_count)]
        s = PlcaSettings(tuple(burst), order)
        if gov.accepts(s):
            return s
    return None


@dataclass
class PlanPolicy(Policy):
    """Rule table from network state to burst counts, applied through the governor.

    predictor: "tinyml", "ewma" (reactive, no ML), or a callable giving the next
    state, which is how wrong predictions are injected in tests.
    """

    cfg: PLCAConfig
    base: PlcaSettings
    governor: PlanGovernor | None = None
    predictor: object = "ewma"
    model: object = None
    window_s: float = 5e-3
    hysteresis: int = 2
    high_burst: int = 1
    congested_burst: int = 3
    state: int = 0
    pending: int = 0
    pending_count: int = 0
    active: PlcaSettings | None = None
    overrides: int = 0
    proposals: int = 0
    changes: int = 0
    monitor: NetworkMonitor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__(self.cfg)
        self.monitor = NetworkMonitor(self.cfg, self.window_s)
        self.model = getattr(self.model, "tree", self.model)
        self.active = self.base

    def _next_state(self, feat: dict[str, float]) -> int:
        if callable(self.predictor):
            return int(self.predictor(feat))
        if self.predictor == "tinyml":
            return int(self.model.predict(np.array([[feat[f] for f in FEATURES]]))[0])
        return int(state_of([feat["util_ewma"]], [feat["queue_total"]])[0])

    def _proposal(self, obs: Observation) -> PlcaSettings:
        """Rule table: raise bursts of backlogged nodes as the state worsens."""
        extra = {2: self.high_burst, 3: self.congested_burst}.get(self.state, 0)
        burst = [b + extra if q >= 2 else b for b, q in zip(self.base.burst, obs.queue_lengths)]
        return PlcaSettings(tuple(burst), self.base.order)

    def plan(self, obs: Observation) -> CyclePlan:
        feat = self.monitor.observe(obs)
        if feat is not None:
            new = self._next_state(feat)
            self.pending_count = self.pending_count + 1 if new == self.pending else 1
            self.pending = new
            if self.pending_count >= self.hysteresis and new != self.state:
                self.state = new
            self.proposals += 1
            wanted = self._proposal(obs)
            if self.governor is None or self.governor.accepts(wanted):
                if wanted != self.active:
                    self.changes += 1
                self.active = wanted
            else:
                self.overrides += 1
        s = self.active
        return CyclePlan([1 + b for b in s.burst], order=s.order)

    def frames_per_to(self, obs: Observation) -> list[int]:
        return self.plan(obs).limits

    @property
    def override_rate(self) -> float:
        return self.overrides / self.proposals if self.proposals else 0.0
