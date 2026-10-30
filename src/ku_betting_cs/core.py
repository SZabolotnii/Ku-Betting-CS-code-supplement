"""Exact quadratic-envelope e-factors.

For a conditionally centered score ``phi`` with
``E(phi phi.T | past) <= Fbar`` and a predictable vector ``lambda``, set
``h = lambda.T phi`` and ``v = lambda.T Fbar lambda``.  Then

    (1 + h + h**2 / 2) / (1 + v / 2)

is nonnegative and has conditional expectation at most one.  The functions
below evaluate this factor in log space and implement the score banks used by
the registered experiments.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def moment_matrix(mu3: float, mu4: float, variance: float = 1.0) -> np.ndarray:
    """Gram matrix for ``(Y, Y^2-variance)`` from central moments."""
    return np.array(
        [[variance, mu3], [mu3, mu4 - variance**2]], dtype=float
    )


def score_direction(mu3: float, mu4: float, method: str) -> np.ndarray:
    """Return a unit-null-variance local location direction.

    ``polynomial`` uses the F^{-1} e1 direction.  ``linear`` and ``catoni``
    use the standardized first-order score.  All returned scores have null
    second moment one, so a common scalar bank is side-information matched.
    """
    if method == "polynomial":
        f = moment_matrix(mu3, mu4)
        raw = np.linalg.solve(f, np.array([1.0, 0.0]))
        return raw / np.sqrt(raw @ f @ raw)
    if method in {"linear", "catoni"}:
        return np.array([1.0, 0.0])
    raise ValueError(f"unknown method: {method}")


def score(x: np.ndarray, m: float | np.ndarray, mu3: float, mu4: float,
          method: str) -> np.ndarray:
    """Evaluate the unit-variance score at candidate location ``m``."""
    y = np.asarray(x) - np.asarray(m)
    if method == "polynomial":
        d = score_direction(mu3, mu4, method)
        return d[0] * y + d[1] * (y * y - 1.0)
    if method in {"linear", "catoni"}:
        return y
    raise ValueError(f"unknown method: {method}")


def _log_q(u: np.ndarray) -> np.ndarray:
    """Stable log(1+u+u^2/2); the quadratic is strictly positive."""
    u = np.asarray(u, dtype=float)
    # Direct evaluation is accurate near zero.  The scaled branch prevents
    # overflow for exceptionally large observations in heavy-tail stress runs.
    out = np.empty_like(u)
    moderate = np.abs(u) < 1.0e150
    out[moderate] = np.log1p(u[moderate] + 0.5 * u[moderate] ** 2)
    a = np.abs(u[~moderate])
    out[~moderate] = 2.0 * np.log(a) - np.log(2.0)
    return out


def catoni_influence(u: np.ndarray) -> np.ndarray:
    """The Wang--Ramdas Catoni influence function."""
    u = np.asarray(u, dtype=float)
    return np.where(u >= 0.0, _log_q(u), -_log_q(-u))


def log_factor(z: np.ndarray, scales: np.ndarray, method: str) -> np.ndarray:
    """Log factors for a unit-variance scalar score and a scale bank."""
    u = np.asarray(z)[..., None] * np.asarray(scales)
    v = np.asarray(scales) ** 2
    if method in {"polynomial", "linear"}:
        return _log_q(u) - np.log1p(v / 2.0)
    if method == "catoni":
        return catoni_influence(u) - v / 2.0
    raise ValueError(f"unknown method: {method}")


def log_quadratic_factor(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate ``log{(1+h+h^2/2)/(1+v/2)}`` elementwise."""
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    if np.any(v < 0.0):
        raise ValueError("the variance proxy v must be nonnegative")
    return _log_q(h) - np.log1p(v / 2.0)


def predictable_ftrl_path(
    z: np.ndarray,
    regularization: float,
    cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic-FTRL e-process for a unit-second-moment scalar score.

    At time ``t`` the stake is

    ``clip(sum(z[:t-1]) / (regularization + t - 1), -cap, cap)``.

    It therefore depends only on past scores.  The regularization and cap are
    numerical/tuning choices, not positivity requirements of the envelope.
    The final axis is interpreted as time; leading axes index paths.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 0:
        raise ValueError("z must have a time axis")
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    if cap <= 0.0:
        raise ValueError("cap must be positive")
    prefix = np.cumsum(z, axis=-1)
    stakes = np.zeros_like(z)
    if z.shape[-1] > 1:
        denominator = regularization + np.arange(1, z.shape[-1], dtype=float)
        stakes[..., 1:] = np.clip(
            prefix[..., :-1] / denominator,
            -cap,
            cap,
        )
    increments = log_quadratic_factor(stakes * z, stakes * stakes)
    return np.cumsum(increments, axis=-1), stakes


def log_mixture_path(
    x: np.ndarray,
    m: float,
    mu3: float,
    mu4: float,
    method: str,
    scales: np.ndarray,
) -> np.ndarray:
    """Equal-weight mixture e-process, returned at every time point.

    ``x`` may have shape ``(n,)`` or ``(replications, n)``.  The last axis is
    time.  The output has the same leading axes and a final time axis.
    """
    z = score(x, m, mu3, mu4, method)
    increments = log_factor(z, scales, method)
    log_wealth = np.cumsum(increments, axis=-2)
    return logsumexp(log_wealth, axis=-1) - np.log(len(scales))


def terminal_log_mixture_grid(
    x: np.ndarray,
    candidates: np.ndarray,
    mu3: float,
    mu4: float,
    method: str,
    scales: np.ndarray,
    chunk: int = 24,
) -> np.ndarray:
    """Evaluate terminal mixture e-values on a location grid."""
    x = np.asarray(x, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    out = np.empty(candidates.size, dtype=float)
    for start in range(0, candidates.size, chunk):
        stop = min(start + chunk, candidates.size)
        m = candidates[start:stop, None]
        z = score(x[None, :], m, mu3, mu4, method)
        logw = np.sum(log_factor(z, scales, method), axis=-2)
        out[start:stop] = logsumexp(logw, axis=-1) - np.log(len(scales))
    return out


def cs_width_from_grid(candidates: np.ndarray, loge: np.ndarray,
                       alpha: float) -> tuple[float, bool]:
    """Width of the convex hull of nonrejected grid points.

    The boolean reports whether the accepted set touched a grid endpoint.
    """
    accepted = np.asarray(loge) < np.log(1.0 / alpha)
    if not np.any(accepted):
        return 0.0, False
    idx = np.flatnonzero(accepted)
    touched = bool(idx[0] == 0 or idx[-1] == len(candidates) - 1)
    return float(candidates[idx[-1]] - candidates[idx[0]]), touched


def crossing_times(loge: np.ndarray, alpha: float) -> np.ndarray:
    """First threshold crossing; ``n+1`` denotes no crossing by the horizon."""
    hit = np.asarray(loge) >= np.log(1.0 / alpha)
    first = np.argmax(hit, axis=-1) + 1
    return np.where(np.any(hit, axis=-1), first, hit.shape[-1] + 1)
