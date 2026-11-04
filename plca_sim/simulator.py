"""Cycle-accurate simulator of PLCA on a 10BASE-T1S multidrop segment.

Model (simplified from IEEE 802.3cg clause 148):
  * Node 0 (coordinator) sends a BEACON to start each cycle.
  * Transmit opportunities (TOs) follow in node-ID order 0, 1, ..., N-1.
  * A node with a queued frame transmits at the start of its TO; the TO then
    lasts for the frame's bus time. With burst mode / an adaptive policy a node
    may send several frames back to back in one TO.
  * A node with nothing to send lets its TO expire after `to_timer_bt`.
  * Within a node, critical frames are served before non-critical ones.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Iterable

from .config import PLCAConfig
from .policy import ConventionalPolicy, Observation, Policy
from .traffic import Frame


@dataclass
class FrameRecord:
    node: int
    arrival: float
    start: float
    end: float
    size: int
    critical: bool
    deadline: float | None


@dataclass
class SimResult:
    cfg: PLCAConfig
    duration_bt: float
    offered: int
    records: list[FrameRecord] = field(default_factory=list)
    dropped: list[Frame] = field(default_factory=list)
    tos_total: int = 0
    tos_used: int = 0
    busy_bt: float = 0.0
    queue_sum: list[int] = field(default_factory=list)
    queue_samples: list[int] = field(default_factory=list)
    cycle_lengths: list[float] = field(default_factory=list)

    @property
    def backlog(self) -> int:
        """Frames still queued (neither sent nor dropped) at the end."""
        return self.offered - len(self.records) - len(self.dropped)


def simulate(
    frames: Iterable[Frame],
    cfg: PLCAConfig,
    duration_bt: float,
    policy: Policy | None = None,
) -> SimResult:
    policy = policy or ConventionalPolicy(cfg)
    n_nodes = cfg.node_count

    pending: list[list[Frame]] = [[] for _ in range(n_nodes)]
    for f in sorted(frames, key=lambda f: f.arrival):
        if not 0 <= f.node < n_nodes:
            raise ValueError(f"frame for node {f.node} but only {n_nodes} nodes")
        pending[f.node].append(f)
    ptr = [0] * n_nodes
    ready: list[list[tuple]] = [[] for _ in range(n_nodes)]
    n_critical = [0] * n_nodes
    seq = itertools.count()

    res = SimResult(
        cfg=cfg,
        duration_bt=duration_bt,
        offered=sum(len(p) for p in pending),
        queue_sum=[0] * n_nodes,
        queue_samples=[0] * n_nodes,
    )

    def enqueue(n: int, now: float) -> None:
        lst, i, q = pending[n], ptr[n], ready[n]
        while i < len(lst) and lst[i].arrival <= now:
            f = lst[i]
            i += 1
            if len(q) >= cfg.max_queue:
                res.dropped.append(f)
                continue
            heapq.heappush(q, (0 if f.critical else 1, f.arrival, next(seq), f))
            if f.critical:
                n_critical[n] += 1
        ptr[n] = i

    t = 0.0
    cycle = 0
    last_cycle = float(cfg.empty_cycle_bt)
    while t < duration_bt:
        cycle_start = t
        for n in range(n_nodes):
            enqueue(n, t)
        obs = Observation(
            cycle=cycle,
            now_bt=t,
            queue_lengths=[len(q) for q in ready],
            critical_waiting=list(n_critical),
            last_cycle_bt=last_cycle,
        )
        limits = policy.frames_per_to(obs)

        t += cfg.beacon_bt
        for n in range(n_nodes):
            enqueue(n, t)
            q = ready[n]
            res.queue_sum[n] += len(q)
            res.queue_samples[n] += 1
            res.tos_total += 1
            if not q:
                t += cfg.to_timer_bt
                continue
            res.tos_used += 1
            sent = 0
            while q and sent < max(1, limits[n]):
                _, _, _, f = heapq.heappop(q)
                if f.critical:
                    n_critical[n] -= 1
                tx = cfg.wire_bits(f.size)
                res.records.append(
                    FrameRecord(n, f.arrival, t, t + tx, f.size, f.critical, f.deadline)
                )
                t += tx
                res.busy_bt += tx
                sent += 1
                enqueue(n, t)

        last_cycle = t - cycle_start
        res.cycle_lengths.append(last_cycle)
        cycle += 1

    return res
