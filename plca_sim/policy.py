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


class Policy:
    def __init__(self, cfg: PLCAConfig) -> None:
        self.cfg = cfg

    def frames_per_to(self, obs: Observation) -> list[int]:
        """Return, per node, the maximum number of frames for this cycle (>= 1)."""
        raise NotImplementedError


class ConventionalPolicy(Policy):
    """Standard PLCA: static round-robin, 1 + max_bc frames per TO for all nodes."""

    def frames_per_to(self, obs: Observation) -> list[int]:
        return [1 + self.cfg.max_bc] * self.cfg.node_count
