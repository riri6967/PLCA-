"""Self-adaptive PLCA: policy engine + deterministic safety governor.

Framework (research plan, section 9):

    PLCA monitoring -> feature extraction -> TinyML prediction
        -> adaptive policy engine -> deterministic safety governor -> PLCA bus

* The **adaptive policy engine** turns the predicted network state into a
  per-node transmission plan: under high load or congestion, nodes with a
  backlog may send several frames per TO (a burst), so bursts are cleared in
  one cycle instead of many.
* The **safety governor** checks every plan before it reaches the bus. It
  never trusts the prediction: whatever the policy asks for, it enforces

    1. a fixed number of critical frames per TO for each critical node, and
    2. a per-cycle budget for non-critical bus time,

  which together give a worst-case cycle length and therefore a provable
  latency bound for every safety-critical flow (see `SafetyGovernor`).
  Without critical flows it caps the cycle at conventional PLCA's own
  worst case, so adaptation can never make the worst-case wait longer.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .config import PLCAConfig
from .monitor import FEATURES, NetworkMonitor
from .policy import CyclePlan, Observation, Policy
from .tinyml import state_of
from .traffic import CriticalFlow

MAX_FRAME_BYTES = 1518


class GovernorInfeasible(ValueError):
    """The critical flows cannot be guaranteed on this segment."""


class SafetyGovernor:
    """Deterministic guard between the policy engine and the bus.

    Let F = beacon + N * TO (fixed cycle overhead), c_i the bus time of one
    frame of critical flow i, P_i its period and D the tightest deadline
    (times `margin`). The governor fixes

        L_i = ceil(D / P_i)                       critical frames per TO
        B   = D - F - sum_i L_i c_i - max_i L_i c_i   non-critical budget

    The budget B is counted from the start of each critical node's TO (the
    critical nodes are "anchors"). Between the starts of two consecutive TOs
    of a critical node there is one BEACON, one TO of every node, at most
    L_j critical frames of every critical node j and at most B of
    non-critical traffic, so the gap is at most  C = F + sum_i L_i c_i + B.
    At most ceil(C / P_i) <= L_i of its frames arrive in such a gap, so its
    queue is cleared at every visit and each critical frame is delivered
    within  C + L_i c_i <= D.

    Without critical flows every node is an anchor and B = N * (max frame),
    so the gap between two TOs of any node never exceeds conventional
    PLCA's worst case.
    """

    def __init__(
        self,
        cfg: PLCAConfig,
        flows: Sequence[CriticalFlow] = (),
        margin: float = 0.95,
    ) -> None:
        self.cfg = cfg
        self.flows = tuple(flows)
        self.max_frame_bt = cfg.wire_bits(MAX_FRAME_BYTES)
        fixed = cfg.beacon_bt + cfg.node_count * cfg.to_timer_bt
        self.crit_limit: dict[int, int] = {}
        self._next_reserved = 0

        if not self.flows:
            # No hard deadlines: never allow a longer gap between two TOs of
            # a node than conventional PLCA's worst case (one maximum-size
            # frame from every node).
            self.anchors = tuple(range(cfg.node_count))
            self.nc_budget_bt = float(cfg.node_count * self.max_frame_bt)
            self.worst_case_gap_bt = fixed + self.nc_budget_bt
            self.latency_bound_bt: dict[int, float] = {}
            return

        deadline = min(cfg.s_to_bt(f.deadline_s) for f in self.flows) * margin
        crit_bt: dict[int, float] = {}
        for f in self.flows:
            period = cfg.s_to_bt(f.period_s)
            n = math.ceil(deadline / period)
            self.crit_limit[f.node] = self.crit_limit.get(f.node, 0) + n
            crit_bt[f.node] = crit_bt.get(f.node, 0.0) + n * cfg.wire_bits(f.size)
        total_crit = sum(crit_bt.values())
        self.nc_budget_bt = deadline - fixed - total_crit - max(crit_bt.values())
        if self.nc_budget_bt < self.max_frame_bt:
            raise GovernorInfeasible(
                "deadline too tight: the non-critical budget cannot fit one maximum-size frame"
            )
        self.anchors = tuple(sorted(crit_bt))
        self.worst_case_gap_bt = fixed + total_crit + self.nc_budget_bt
        self.latency_bound_bt = {
            node: self.worst_case_gap_bt + bt for node, bt in crit_bt.items()
        }

    def enforce(self, plan: CyclePlan, obs: Observation) -> CyclePlan:
        limits = list(plan.limits)
        for node, n in self.crit_limit.items():
            limits[node] = n
        budget = self.nc_budget_bt
        if plan.nc_budget_bt is not None:
            budget = min(budget, plan.nc_budget_bt)
        return CyclePlan(
            limits=limits,
            nc_budget_bt=budget,
            anchors=self.anchors,
            reserved_node=self._pick_reserved(obs),
            reserve_bt=self.max_frame_bt,
        )

    def _pick_reserved(self, obs: Observation) -> int | None:
        """Rotate a reservation over nodes with non-critical backlog.

        A fairness aid (nodes late in the TO order are not starved by the
        budget); it plays no part in the safety bound."""
        n = self.cfg.node_count
        start = self._next_reserved
        for k in range(n):
            node = (start + k) % n
            if obs.queue_lengths[node] > obs.critical_waiting[node]:
                self._next_reserved = (node + 1) % n
                return node
        return None


class AdaptivePolicy(Policy):
    """Policy engine driven by the predicted (or measured) network state.

    mode:
      "tinyml"   - use the TinyML model's prediction of the next 20 ms
      "reactive" - use the currently measured state (no prediction)
      "off"      - no adaptation (1 frame per TO), governor only
    """

    def __init__(
        self,
        cfg: PLCAConfig,
        governor: SafetyGovernor | None = None,
        model=None,
        mode: str = "tinyml",
        window_s: float = 5e-3,
        max_burst: int = 8,
    ) -> None:
        super().__init__(cfg)
        if mode not in ("tinyml", "reactive", "off"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "tinyml" and model is None:
            raise ValueError("mode 'tinyml' needs a trained model")
        self.governor = governor
        self.model = getattr(model, "tree", model)  # TrainedModel or sklearn tree
        self.mode = mode
        self.max_burst = max_burst
        self.monitor = NetworkMonitor(cfg, window_s)
        self.state = 0
        self.state_counts = [0, 0, 0, 0]

    def _update_state(self, obs: Observation) -> None:
        feat = self.monitor.observe(obs)
        if feat is None or self.mode == "off":
            return
        if self.mode == "tinyml":
            x = np.array([[feat[f] for f in FEATURES]])
            self.state = int(self.model.predict(x)[0])
        else:
            self.state = int(state_of([feat["util_ewma"]], [feat["queue_total"]])[0])
        self.state_counts[self.state] += 1

    def frames_per_to(self, obs: Observation) -> list[int]:
        q = obs.queue_lengths
        if self.state == 3:  # congested: let backlogged nodes clear bursts
            return [min(x, self.max_burst) if x >= 2 else 1 for x in q]
        if self.state == 2:  # high: moderate bursts for long queues
            return [2 if x >= 3 else 1 for x in q]
        return [1] * self.cfg.node_count  # low / medium: conventional PLCA

    def plan(self, obs: Observation) -> CyclePlan:
        self._update_state(obs)
        plan = CyclePlan(self.frames_per_to(obs))
        if self.governor is not None:
            plan = self.governor.enforce(plan, obs)
        return plan
