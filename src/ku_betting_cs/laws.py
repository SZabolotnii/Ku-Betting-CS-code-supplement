"""Standardized laws used by the registered BET gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Law:
    name: str
    mu3: float
    mu4: float
    sampler: Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]
    description: str

    @property
    def g2(self) -> float:
        return 1.0 - self.mu3**2 / (self.mu4 - 1.0)

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        shape = (size,) if isinstance(size, int) else tuple(size)
        return self.sampler(rng, shape)


def _scipy_law(name: str, dist, description: str) -> Law:
    mean, var, skew, excess = dist.stats(moments="mvsk")
    mean, var, skew, excess = map(float, (mean, var, skew, excess))

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        return (dist.rvs(size=shape, random_state=rng) - mean) / np.sqrt(var)

    return Law(name, skew, excess + 3.0, sampler, description)


def _mixture_law() -> Law:
    p, a = 0.2, 2.5
    var = 1.0 + a * a * p * (1.0 - p)
    mu3_raw = a**3 * p * (1.0 - p) * (1.0 - 2.0 * p)
    mu4_raw = (
        3.0
        + 6.0 * a * a * p * (1.0 - p)
        + a**4 * p * (1.0 - p) * ((1.0 - p) ** 3 + p**3)
    )

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        b = rng.binomial(1, p, size=shape)
        return (rng.normal(size=shape) + a * (b - p)) / np.sqrt(var)

    return Law(
        "skew_mixture",
        mu3_raw / var**1.5,
        mu4_raw / var**2,
        sampler,
        "standardized N(0,1) plus a centered Bernoulli jump",
    )


def _ziln_law(nonzero: float = 0.25, log_sigma: float = 1.0) -> Law:
    raw = np.array(
        [1.0, *[nonzero * np.exp(0.5 * (k * log_sigma) ** 2) for k in range(1, 5)]]
    )
    mean = raw[1]
    var = raw[2] - mean**2
    mu3 = raw[3] - 3 * mean * raw[2] + 2 * mean**3
    mu4 = raw[4] - 4 * mean * raw[3] + 6 * mean**2 * raw[2] - 3 * mean**4

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        active = rng.random(shape) < nonzero
        x = active * rng.lognormal(0.0, log_sigma, size=shape)
        return (x - mean) / np.sqrt(var)

    return Law(
        f"ziln_p{int(100 * nonzero):02d}",
        mu3 / var**1.5,
        mu4 / var**2,
        sampler,
        f"standardized zero-inflated lognormal, nonzero={nonzero:g}",
    )


_LAWS = {
    "gaussian": _scipy_law("gaussian", stats.norm(), "standard normal control"),
    "gamma2": _scipy_law("gamma2", stats.gamma(a=2.0), "standardized Gamma(shape=2)"),
    "lognormal04": _scipy_law(
        "lognormal04", stats.lognorm(s=0.4), "standardized lognormal(log-sd=0.4)"
    ),
    "pareto5": _scipy_law(
        "pareto5", stats.pareto(b=5.0), "standardized Pareto(shape=5; finite fourth moment)"
    ),
    "skew_mixture": _mixture_law(),
    "ziln_p25": _ziln_law(0.25, 1.0),
    "ziln_p60": _ziln_law(0.60, 0.7),
}


def law_names() -> tuple[str, ...]:
    return tuple(_LAWS)


def get_law(name: str) -> Law:
    return _LAWS[name]
