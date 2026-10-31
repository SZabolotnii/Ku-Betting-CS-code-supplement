"""Moment information transferred from an independent calibration sample.

The location score ``(X-m, (X-m)^2-c)`` is centered only when ``c`` equals the
true variance.  When an independent reference sample places the variance in
``[c-r, c+r]``, the conditional mean of the second component is some
``Delta`` with ``|Delta| <= r``.  The centering-robust factor

    (1 + h - |lambda_2| r + h**2 / 2) / (1 + v / 2),
    h = lambda.T phi,  v = lambda.T Fbar lambda,

has conditional expectation at most one whenever
``Fbar >= Cov(phi) + r^2 e2 e2.T`` and is strictly positive whenever
``|lambda_2| r < 1/2``.  This module builds the calibration region, the
matrix ``Fbar``, the horizon-matched score bank and the factor itself.

Calibration uses only the reference sample.  For a variable ``Y`` with
``Var(Y) <= v``, the Catoni influence function ``psi`` satisfies
``exp(psi(x)) <= 1 + x + x^2/2``; hence ``prod exp(psi(lam (Y_i-EY))) /
(1 + lam^2 v / 2)`` has expectation at most one and Markov's inequality gives
a finite-sample one-sided bound.  Because ``psi`` is nondecreasing, each bound
is the root of a monotone equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from .core import _log_q


def _psi(u: np.ndarray) -> np.ndarray:
    """Catoni influence function ``sign(u) log(1+|u|+u^2/2)``."""
    a = np.abs(u)
    return np.sign(u) * np.log1p(a + 0.5 * a * a)


def catoni_bound(y: np.ndarray, variance_bound: float, delta: float,
                 side: str) -> float:
    """Finite-sample one-sided confidence bound for ``E(Y)``.

    ``y`` holds i.i.d. draws with ``Var(Y) <= variance_bound``.  ``side`` is
    ``"lower"`` or ``"upper"``; the bound fails with probability at most
    ``delta``.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if side not in {"lower", "upper"}:
        raise ValueError(f"unknown side: {side}")
    if variance_bound <= 0.0 or not 0.0 < delta < 1.0 or n < 2:
        raise ValueError("need a positive variance bound, 0<delta<1 and n>=2")
    lam = np.sqrt(2.0 * np.log(1.0 / delta) / (n * variance_bound))
    threshold = n * np.log1p(0.5 * lam * lam * variance_bound) + np.log(1.0 / delta)
    target = threshold if side == "lower" else -threshold

    def equation(theta: float) -> float:
        # Decreasing in theta because psi is nondecreasing.
        return float(np.sum(_psi(lam * (y - theta)))) - target

    center = float(np.mean(y))
    step = 4.0 * np.sqrt(variance_bound * np.log(1.0 / delta) / n) + 1e-12
    lo, hi = center - step, center + step
    while equation(lo) < 0.0:
        lo -= step
        step *= 2.0
    while equation(hi) > 0.0:
        hi += step
        step *= 2.0
    return float(brentq(equation, lo, hi, xtol=1e-13, rtol=1e-12))


@dataclass(frozen=True)
class MomentRegion:
    """Calibration region for a centered error with known zero mean.

    ``variance`` lies in ``[center-radius, center+radius]``, ``mu3`` in
    ``[mu3_center-mu3_radius, mu3_center+mu3_radius]`` and ``mu4`` below
    ``mu4_upper``.  ``fbar`` dominates ``Cov(phi)+radius^2 e2 e2.T`` for every
    moment triple in the region.
    """

    center: float
    radius: float
    mu3_center: float
    mu3_radius: float
    mu4_upper: float
    crude_mu4_upper: float
    fbar: np.ndarray

    @property
    def variance_upper(self) -> float:
        return self.center + self.radius

    @property
    def variance_lower(self) -> float:
        return max(self.center - self.radius, 0.0)

    def covers(self, variance: float, mu3: float, mu4: float) -> bool:
        return bool(
            abs(variance - self.center) <= self.radius
            and abs(mu3 - self.mu3_center) <= self.mu3_radius
            and mu4 <= self.mu4_upper
        )

    @property
    def g2_bar(self) -> float:
        f = self.fbar
        return float(1.0 - f[0, 1] ** 2 / (f[0, 0] * f[1, 1]))


def _split(eps: np.ndarray, crude_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    k = int(round(crude_fraction * eps.size))
    if not 0 < k < eps.size:
        raise ValueError("crude_fraction leaves an empty split")
    return eps[:k], eps[k:]


def calibrate_moments(eps: np.ndarray, eighth_moment_bound: float,
                      deltas: dict, crude_fraction: float) -> MomentRegion:
    """Moment region from reference errors ``eps`` with known zero mean.

    The only tail assumption is ``E(eps^8) <= eighth_moment_bound``.  A crude
    upper bound for ``E(eps^4)`` from the first split supplies the variance
    proxies used on the second split, so every bound below is valid
    conditionally on the first split.  The four failure probabilities in
    ``deltas`` (``crude_mu4``, ``variance``, ``mu3``, ``mu4``) add up to the
    total calibration error.
    """
    eps = np.asarray(eps, dtype=float)
    k8 = float(eighth_moment_bound)
    first, second = _split(eps, crude_fraction)
    crude_mu4 = catoni_bound(first**4, k8, deltas["crude_mu4"], "upper")
    crude_mu4 = max(crude_mu4, 1e-12)

    y2 = second**2
    half = deltas["variance"] / 2.0
    var_lo = catoni_bound(y2, crude_mu4, half, "lower")
    var_hi = catoni_bound(y2, crude_mu4, half, "upper")
    center, radius = 0.5 * (var_lo + var_hi), 0.5 * (var_hi - var_lo)

    # E(eps^6) <= sqrt(E eps^4 E eps^8) bounds Var(eps^3) on the first-split event.
    v3 = np.sqrt(crude_mu4 * k8)
    half3 = deltas["mu3"] / 2.0
    m3_lo = catoni_bound(second**3, v3, half3, "lower")
    m3_hi = catoni_bound(second**3, v3, half3, "upper")
    mu3_center, mu3_radius = 0.5 * (m3_lo + m3_hi), 0.5 * (m3_hi - m3_lo)

    mu4_upper = catoni_bound(second**4, k8, deltas["mu4"], "upper")
    fbar = dominating_matrix(center, radius, mu3_center, mu3_radius, mu4_upper)
    return MomentRegion(center, radius, mu3_center, mu3_radius, mu4_upper,
                        crude_mu4, fbar)


def calibrate_variance_upper(eps: np.ndarray, eighth_moment_bound: float,
                             deltas: dict, crude_fraction: float) -> float:
    """Upper variance bound for the first-order comparator (same chain)."""
    eps = np.asarray(eps, dtype=float)
    first, second = _split(eps, crude_fraction)
    crude_mu4 = catoni_bound(first**4, float(eighth_moment_bound),
                             deltas["crude_mu4"], "upper")
    return catoni_bound(second**2, max(crude_mu4, 1e-12),
                        deltas["variance_upper"], "upper")


# ------------------------------------------------- reference of unknown location
#
# A reference standard of known value is not always available; a long record of
# the in-control process is.  Write ``Y = mu_ref + eps`` with ``mu_ref``
# unknown.  A small first split fixes a centering ``c``; conditionally on it the
# shifted values ``z = Y - c = eps - d`` (``d = c - mu_ref``) are i.i.d., so the
# Catoni bounds apply to the raw moments ``m_k = E z^k``, including ``m_1 = -d``.
# Central moments are polynomials in ``m_1..m_4`` and interval arithmetic
# carries the raw-moment box to a region for ``(sigma^2, mu3, mu4)``.

Interval = tuple[float, float]


def _iadd(*parts: Interval) -> Interval:
    return (sum(p[0] for p in parts), sum(p[1] for p in parts))


def _imul(a: Interval, b: Interval) -> Interval:
    corners = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(corners), max(corners))


def _iscale(k: float, a: Interval) -> Interval:
    return (k * a[0], k * a[1]) if k >= 0.0 else (k * a[1], k * a[0])


def _ipow(a: Interval, k: int) -> Interval:
    lo, hi = a[0] ** k, a[1] ** k
    if k % 2 == 1:
        return (lo, hi)
    floor = 0.0 if a[0] <= 0.0 <= a[1] else min(lo, hi)
    return (floor, max(lo, hi))


def central_from_raw(m1: Interval, m2: Interval, m3: Interval,
                     m4: Interval) -> tuple[Interval, Interval, Interval]:
    """Enclosures of ``sigma^2, mu3, mu4`` from enclosures of raw moments.

    ``sigma^2 = m2 - m1^2``, ``mu3 = m3 - 3 m1 m2 + 2 m1^3`` and
    ``mu4 = m4 - 4 m1 m3 + 6 m1^2 m2 - 3 m1^4``; every term is enclosed
    separately, which is conservative but valid.
    """
    sq = _ipow(m1, 2)
    var = _iadd(m2, _iscale(-1.0, sq))
    mu3 = _iadd(m3, _iscale(-3.0, _imul(m1, m2)), _iscale(2.0, _ipow(m1, 3)))
    mu4 = _iadd(m4, _iscale(-4.0, _imul(m1, m3)), _iscale(6.0, _imul(sq, m2)),
                _iscale(-3.0, _ipow(m1, 4)))
    return (max(var[0], 0.0), var[1]), mu3, mu4


def _three_way(y: np.ndarray, location_fraction: float,
               crude_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ka = int(round(location_fraction * y.size))
    kb = int(round(crude_fraction * y.size))
    if ka < 1 or kb < 2 or ka + kb > y.size - 2:
        raise ValueError("fractions leave an empty split")
    return y[:ka], y[ka:ka + kb], y[ka + kb:]


def _shifted_crude(y: np.ndarray, k8: float, deltas: dict, location_fraction: float,
                   crude_fraction: float) -> tuple[np.ndarray, float, float]:
    """Centering split, then the envelope and crude fourth-moment bound for ``z``.

    Returns the third split shifted by ``c``, the envelope ``K8' >= E z^8`` and
    the crude bound ``U >= E z^4``, all valid on the first two bound events.
    """
    first, second, third = _three_way(y, location_fraction, crude_fraction)
    c = float(np.mean(first))
    z2 = second - c
    # |d| = |c - mu_ref| is bounded from the first split alone, so K8' is fixed
    # before the second split is used.  Var(y) = sigma^2 <= (E eps^8)^(1/4).
    half = deltas["shift"] / 2.0
    v1 = k8 ** 0.25
    d_hi = max(abs(c - catoni_bound(first, v1, half, "lower")),
               abs(c - catoni_bound(first, v1, half, "upper")))
    # Minkowski: (E z^8)^(1/8) <= (E eps^8)^(1/8) + |d|.
    k8_shift = (k8 ** 0.125 + d_hi) ** 8
    crude = catoni_bound(z2**4, k8_shift, deltas["crude_mu4"], "upper")
    return third - c, k8_shift, max(crude, 1e-12)


def calibrate_moments_unknown_center(y: np.ndarray, eighth_moment_bound: float,
                                     deltas: dict, location_fraction: float,
                                     crude_fraction: float) -> MomentRegion:
    """Moment region from a reference record ``y`` of unknown location.

    Only ``E(eps^8) <= eighth_moment_bound`` is assumed for the central error.
    ``deltas`` holds ``shift``, ``crude_mu4``, ``mean``, ``variance``, ``mu3``
    and ``mu4``; they add up to the total calibration error.
    """
    y = np.asarray(y, dtype=float)
    z, k8s, crude = _shifted_crude(y, float(eighth_moment_bound), deltas,
                                   location_fraction, crude_fraction)

    def two_sided(values: np.ndarray, v: float, delta: float) -> Interval:
        return (catoni_bound(values, v, delta / 2.0, "lower"),
                catoni_bound(values, v, delta / 2.0, "upper"))

    # Var(z) <= sqrt(E z^4); Var(z^2) <= E z^4; Var(z^3) <= E z^6 <= sqrt(E z^4 E z^8).
    m1 = two_sided(z, np.sqrt(crude), deltas["mean"])
    m2 = two_sided(z**2, crude, deltas["variance"])
    m3 = two_sided(z**3, np.sqrt(crude * k8s), deltas["mu3"])
    m4 = (0.0, catoni_bound(z**4, k8s, deltas["mu4"], "upper"))
    var, mu3, mu4 = central_from_raw(m1, m2, m3, m4)

    center, radius = 0.5 * (var[0] + var[1]), 0.5 * (var[1] - var[0])
    mu3_center, mu3_radius = 0.5 * (mu3[0] + mu3[1]), 0.5 * (mu3[1] - mu3[0])
    fbar = dominating_matrix(center, radius, mu3_center, mu3_radius, mu4[1])
    return MomentRegion(center, radius, mu3_center, mu3_radius, mu4[1], crude, fbar)


def calibrate_variance_upper_unknown_center(y: np.ndarray, eighth_moment_bound: float,
                                            deltas: dict, location_fraction: float,
                                            crude_fraction: float) -> float:
    """First-order comparator: ``sigma^2 = m2 - m1^2 <= m2`` (same chain)."""
    y = np.asarray(y, dtype=float)
    z, _, crude = _shifted_crude(y, float(eighth_moment_bound), deltas,
                                 location_fraction, crude_fraction)
    return catoni_bound(z**2, crude, deltas["variance_upper"], "upper")


def dominating_matrix(center: float, radius: float, mu3_center: float,
                      mu3_radius: float, mu4_upper: float) -> np.ndarray:
    """``Fbar`` with ``Fbar >= Cov(phi) + radius^2 e2 e2.T`` on the region.

    For variance ``s`` in ``[center-radius, center+radius]``, third moment
    within ``mu3_radius`` of ``mu3_center`` and ``mu4 <= mu4_upper``,
    ``Cov(phi)=[[s, mu3], [mu3, mu4-s^2]]``.  The off-diagonal uncertainty is
    absorbed by diagonal increments ``a, b`` with ``a b = mu3_radius^2``;
    ``a`` minimizes the local width proxy ``f11 - f12^2/f22``, which gives
    ``a = r3 (|c| - r3) / T`` with ``T`` the unincremented ``f22``.  When
    ``|c| <= r3`` the minimizer is at the boundary and the score degenerates
    to the first-order one; a tiny ``a`` keeps the matrix finite.
    """
    s_hi = center + radius
    s_lo = max(center - radius, 0.0)
    t22 = max(mu4_upper - s_lo**2, 1e-12) + radius**2
    r3, c = float(mu3_radius), float(mu3_center)
    if r3 > 0.0:
        a = max(r3 * (abs(c) - r3) / t22, 1e-9 * s_hi)
        b = r3 * r3 / a
    else:
        a = b = 0.0
    return np.array([[s_hi + a, c], [c, t22 + b]], dtype=float)


def width_ratio(g2: float, rho: float | np.ndarray) -> np.ndarray:
    """Local half-width of the best calibrated direction over the linear one.

    ``u(rho) = (1-g) rho + sqrt(g (1 - (1-g) rho^2))`` for ``rho <= 1`` and
    one afterwards; ``u(0)=sqrt(g)`` and ``u(1)=1``.
    """
    rho = np.asarray(rho, dtype=float)
    inside = (1.0 - g2) * rho + np.sqrt(
        np.clip(g2 * (1.0 - (1.0 - g2) * rho**2), 0.0, None)
    )
    return np.where(rho >= 1.0, 1.0, inside)


def robust_bank(fbar: np.ndarray, radius: float, scales: np.ndarray,
                penalty_cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizon-matched stakes for the centering-robust factor.

    Each signed scale ``s`` of the first-order bank targets the half-width
    ``w = s sqrt(f11) u(rho_s)``, ``rho_s = radius sqrt(f11)/(|f12| |s|)``.
    Its stake is ``Fbar^{-1}(w, y)``, where ``y`` is the point of
    ``[-radius, radius]`` closest to ``f12 w / f11``; this is the minimax
    direction for that target.  With ``radius=0`` and ``Fbar=F`` the bank is
    exactly the oracle polynomial bank.  Stakes whose penalty
    ``|lambda_2| radius`` would reach ``penalty_cap < 1/2`` are shrunk.
    """
    f11, f12 = float(fbar[0, 0]), float(fbar[0, 1])
    g2_bar = 1.0 - f12**2 / (f11 * float(fbar[1, 1]))
    lambdas, penalties = [], []
    for s in np.asarray(scales, dtype=float):
        if abs(f12) > 1e-14:
            rho = radius * np.sqrt(f11) / (abs(f12) * abs(s))
        else:
            rho = np.inf
        w = s * np.sqrt(f11) * float(width_ratio(g2_bar, rho))
        y = np.sign(f12 * w) * min(abs(f12 * w) / f11, radius)
        lam = np.linalg.solve(fbar, np.array([w, y]))
        pen = abs(lam[1]) * radius
        if pen >= penalty_cap:
            lam = lam * (penalty_cap / pen)
            pen = penalty_cap
        lambdas.append(lam)
        penalties.append(pen)
    return np.asarray(lambdas), np.asarray(penalties)


def log_robust_factor(x: np.ndarray, m, center: float, lambdas: np.ndarray,
                      penalties: np.ndarray, fbar: np.ndarray) -> np.ndarray:
    """``log{(q(h)-penalty)/(1+v/2)}`` with a trailing bank axis."""
    y = np.asarray(x, dtype=float) - np.asarray(m, dtype=float)
    h = y[..., None] * lambdas[:, 0] + (y * y - center)[..., None] * lambdas[:, 1]
    v = np.einsum("ki,ij,kj->k", lambdas, fbar, lambdas)
    log_q = _log_q(h)
    return log_q + np.log1p(-penalties * np.exp(-log_q)) - np.log1p(v / 2.0)


def robust_log_mixture_path(x: np.ndarray, m, center: float, lambdas: np.ndarray,
                            penalties: np.ndarray, fbar: np.ndarray) -> np.ndarray:
    """Equal-weight mixture over the bank, at every time point."""
    inc = log_robust_factor(x, m, center, lambdas, penalties, fbar)
    return logsumexp(np.cumsum(inc, axis=-2), axis=-1) - np.log(len(penalties))


def robust_terminal_grid(x: np.ndarray, candidates: np.ndarray, center: float,
                         lambdas: np.ndarray, penalties: np.ndarray,
                         fbar: np.ndarray, chunk: int = 24) -> np.ndarray:
    """Terminal mixture log e-values on a location grid."""
    x = np.asarray(x, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    out = np.empty(candidates.size)
    for start in range(0, candidates.size, chunk):
        stop = min(start + chunk, candidates.size)
        inc = log_robust_factor(x[None, :], candidates[start:stop, None],
                                center, lambdas, penalties, fbar)
        out[start:stop] = logsumexp(inc.sum(axis=-2), axis=-1) - np.log(len(penalties))
    return out
