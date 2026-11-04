"""Traffic generators: Poisson, bursty and periodic (safety-critical) flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import PLCAConfig

# Default mix of Ethernet frame sizes in bytes (MAC header through FCS).
DEFAULT_SIZES: tuple[int, ...] = (64, 128, 256, 512, 1024, 1518)
DEFAULT_PROBS: tuple[float, ...] = (0.30, 0.20, 0.15, 0.15, 0.10, 0.10)


@dataclass(frozen=True)
class Frame:
    node: int
    arrival: float  # bit times
    size: int  # bytes
    critical: bool = False
    deadline: float | None = None  # relative deadline in bit times


def mean_wire_bits(cfg: PLCAConfig, sizes: Sequence[int], probs: Sequence[float]) -> float:
    return float(sum(p * cfg.wire_bits(s) for s, p in zip(sizes, probs)))


def _poisson_times(rng: np.random.Generator, rate: float, horizon: float) -> np.ndarray:
    """Arrival times of a Poisson process with `rate` events per BT."""
    if rate <= 0:
        return np.empty(0)
    times: list[np.ndarray] = []
    t = 0.0
    while t < horizon:
        n = int(rate * (horizon - t) * 1.2) + 16
        chunk = t + np.cumsum(rng.exponential(1.0 / rate, n))
        times.append(chunk)
        t = chunk[-1]
    out = np.concatenate(times)
    return out[out < horizon]


def poisson_traffic(
    rng: np.random.Generator,
    cfg: PLCAConfig,
    node: int,
    load: float,
    duration_bt: float,
    sizes: Sequence[int] = DEFAULT_SIZES,
    probs: Sequence[float] = DEFAULT_PROBS,
) -> list[Frame]:
    """Poisson flow whose offered load is `load` (fraction of the bus rate)."""
    rate = load / mean_wire_bits(cfg, sizes, probs)  # frames per BT
    times = _poisson_times(rng, rate, duration_bt)
    sz = rng.choice(sizes, size=len(times), p=probs)
    return [Frame(node, float(t), int(s)) for t, s in zip(times, sz)]


def bursty_traffic(
    rng: np.random.Generator,
    cfg: PLCAConfig,
    node: int,
    load: float,
    duration_bt: float,
    mean_burst: float = 8.0,
    sizes: Sequence[int] = DEFAULT_SIZES,
    probs: Sequence[float] = DEFAULT_PROBS,
) -> list[Frame]:
    """Bursts of frames (geometric burst length) that arrive almost together."""
    burst_rate = load / (mean_burst * mean_wire_bits(cfg, sizes, probs))
    starts = _poisson_times(rng, burst_rate, duration_bt)
    frames: list[Frame] = []
    for t0 in starts:
        k = int(rng.geometric(1.0 / mean_burst))
        for i in range(k):
            t = float(t0) + i  # 1 BT apart: effectively simultaneous
            if t < duration_bt:
                frames.append(Frame(node, t, int(rng.choice(sizes, p=probs))))
    return frames


def periodic_traffic(
    rng: np.random.Generator,
    cfg: PLCAConfig,
    node: int,
    period_s: float,
    size: int,
    duration_bt: float,
    deadline_s: float,
    critical: bool = True,
) -> list[Frame]:
    """Periodic (e.g. safety-critical control) flow with a random phase."""
    period = cfg.s_to_bt(period_s)
    phase = float(rng.uniform(0, period))
    deadline = cfg.s_to_bt(deadline_s)
    times = np.arange(phase, duration_bt, period)
    return [Frame(node, float(t), size, critical, deadline) for t in times]


@dataclass(frozen=True)
class CriticalFlow:
    """A safety-critical periodic flow known at design time."""

    node: int
    period_s: float
    size: int  # bytes
    deadline_s: float
