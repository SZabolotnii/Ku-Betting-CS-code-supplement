"""Small statistical helpers used by the gates."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(successes: int, trials: int, level: float = 0.95) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(level, successes + 1, trials - successes))


def paired_bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                        reps: int = 4000, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return tuple(np.quantile(draws, [tail, 1.0 - tail]).tolist())
