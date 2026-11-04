"""Transmission policies.

A policy is asked at the start of every PLCA cycle how many frames each node
may send in its transmit opportunity. The conventional policy gives every
node the same fixed limit (standard PLCA). The self-adaptive policy
(TinyML prediction + Safety Governor) will subclass `Policy` in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PLCAConfig


@dataclass
class Observation:
    """Network state visible to a policy at the start of a cycle."""

    cycle: int
    now_bt: float
    queue_lengths: list[int]
    critical_waiting: list[int]
    last_cycle_bt: float
    # Cumulative counters since the start of the simulation.
    busy_bt: float = 0.0
    tos_used: int = 0
    tos_total: int = 0
    arrived: int = 0
    dropped: int = 0


@dataclass
class CyclePlan:
    """What the nodes may do in the coming PLCA cycle."""

    # Maximum frames each node may send in its TO (values < 1 count as 1).
    limits: list[int]
    # Bus time (BT) that non-critical frames may use; None = unlimited
    # (conventional PLCA). Without anchors it is counted per cycle. With
    # anchors it is counted, for every anchor node, since the start of that
    # node's last TO, which bounds the gap between two TOs of each anchor.
    nc_budget_bt: float | None = None
    anchors: tuple[int, ...] = ()
    # Node for which `reserve_bt` of the budget is kept free until its TO,
    # so nodes late in the TO order cannot be starved.
    reserved_node: int | None = None
    reserve_bt: float = 0.0


class Policy:
    def __init__(self, cfg: PLCAConfig) -> None:
        self.cfg = cfg

    def frames_per_to(self, obs: Observation) -> list[int]:
        """Return, per node, the maximum number of frames for this cycle (>= 1)."""
        raise NotImplementedError

    def plan(self, obs: Observation) -> CyclePlan:
        """Full plan for the next cycle. Default: frame limits only."""
        return CyclePlan(self.frames_per_to(obs))


class ConventionalPolicy(Policy):
    """Standard PLCA: static round-robin, 1 + max_bc frames per TO for all nodes."""

    def frames_per_to(self, obs: Observation) -> list[int]:
        return [1 + self.cfg.max_bc] * self.cfg.node_count
