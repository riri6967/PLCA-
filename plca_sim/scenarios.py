"""Evaluation scenarios from the research plan (section 11).

  Scenario 1 - low traffic       -> "uniform" at 20-30 % load
  Scenario 2 - increasing traffic -> "uniform" and "heterogeneous" load sweep
  Scenario 3 - bursty traffic    -> "bursty"
  Scenario 4 - mixed criticality -> "mixed"

`load` is the total offered load as a fraction of the 10 Mb/s bus.
"""

from __future__ import annotations

import numpy as np

from .config import PLCAConfig
from .traffic import Frame, bursty_traffic, periodic_traffic, poisson_traffic

SCENARIOS = ("uniform", "heterogeneous", "bursty", "mixed")

# Safety-critical flow used in the mixed-criticality scenario.
CRITICAL_NODES = (1, 2)
CRITICAL_PERIOD_S = 1e-3  # 1 ms control message
CRITICAL_SIZE = 64  # bytes
CRITICAL_DEADLINE_S = 2e-3  # must arrive within 2 ms


def make_traffic(
    scenario: str, load: float, cfg: PLCAConfig, duration_s: float, seed: int
) -> list[Frame]:
    rng = np.random.default_rng(seed)
    horizon = cfg.s_to_bt(duration_s)
    n = cfg.node_count
    frames: list[Frame] = []

    if scenario == "uniform":
        for node in range(n):
            frames += poisson_traffic(rng, cfg, node, load / n, horizon)

    elif scenario == "heterogeneous":
        # A few ECUs (e.g. a camera gateway) carry most of the traffic.
        weights = np.array([2.0 ** -(i % n) for i in range(n)])
        weights /= weights.sum()
        for node in range(n):
            frames += poisson_traffic(rng, cfg, node, load * weights[node], horizon)

    elif scenario == "bursty":
        for node in range(n):
            frames += poisson_traffic(rng, cfg, node, 0.3 * load / n, horizon)
            frames += bursty_traffic(rng, cfg, node, 0.7 * load / n, horizon)

    elif scenario == "mixed":
        crit_load = 0.0
        for node in CRITICAL_NODES:
            flow = periodic_traffic(
                rng, cfg, node, CRITICAL_PERIOD_S, CRITICAL_SIZE, horizon, CRITICAL_DEADLINE_S
            )
            frames += flow
            crit_load += cfg.wire_bits(CRITICAL_SIZE) / cfg.s_to_bt(CRITICAL_PERIOD_S)
        background = [i for i in range(n) if i not in CRITICAL_NODES]
        bg_load = max(load - crit_load, 0.0) / len(background)
        big = (512, 1024, 1518)
        big_p = (0.3, 0.3, 0.4)
        for node in background:
            frames += poisson_traffic(rng, cfg, node, 0.5 * bg_load, horizon, big, big_p)
            frames += bursty_traffic(rng, cfg, node, 0.5 * bg_load, horizon, 6.0, big, big_p)

    else:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")

    return frames
