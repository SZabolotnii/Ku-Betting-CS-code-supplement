import numpy as np

from ku_betting_cs.calibration import (
    calibrate_moments,
    calibrate_moments_unknown_center,
    calibrate_variance_upper_unknown_center,
    catoni_bound,
    central_from_raw,
    dominating_matrix,
    log_robust_factor,
    robust_bank,
    width_ratio,
)
from ku_betting_cs.core import moment_matrix, score_direction
from ku_betting_cs.laws import get_law


def _three_point_law():
    """Centered law with atoms and probabilities, plus its central moments."""
    atoms = np.array([-1.0, 0.5, 3.0])
    p = np.array([0.5, 0.4, 0.1])
    atoms = atoms - p @ atoms
    mom = {k: float(p @ atoms**k) for k in (2, 3, 4)}
    return atoms, p, mom


def test_robust_factor_is_an_e_variable_for_every_admissible_centering() -> None:
    atoms, p, mom = _three_point_law()
    cov = np.array([[mom[2], mom[3]], [mom[3], mom[4] - mom[2] ** 2]])
    radius = 0.2
    fbar = cov + np.diag([0.0, radius**2])
    lam = np.array([0.3, -0.25])
    penalty = abs(lam[1]) * radius
    for delta in np.linspace(-radius, radius, 9):
        center = mom[2] - delta
        logf = log_robust_factor(atoms, 0.0, center, lam[None, :],
                                 np.array([penalty]), fbar)[:, 0]
        expectation = float(p @ np.exp(logf))
        assert expectation <= 1.0 + 1e-12
    # The bound is attained when the centering error has the adverse sign.
    worst = mom[2] - np.sign(lam[1]) * radius
    logf = log_robust_factor(atoms, 0.0, worst, lam[None, :],
                             np.array([penalty]), fbar)[:, 0]
    assert np.isclose(float(p @ np.exp(logf)), 1.0, atol=1e-12)


def test_robust_factor_is_positive_for_extreme_scores() -> None:
    x = np.array([-1e9, -50.0, -1.0, 0.0, 2.0, 1e9])
    fbar = np.array([[1.0, 0.5], [0.5, 3.0]])
    lam = np.array([[0.4, -0.3]])
    logf = log_robust_factor(x, 0.0, 1.0, lam, np.array([0.45]), fbar)
    assert np.all(np.isfinite(logf))


def test_bank_reduces_to_oracle_polynomial_bank_without_radius() -> None:
    law = get_law("gamma2")
    f = moment_matrix(law.mu3, law.mu4)
    scales = np.array([-0.3, -0.1, 0.1, 0.3])
    lambdas, penalties = robust_bank(f, 0.0, scales, 0.45)
    d = score_direction(law.mu3, law.mu4, "polynomial")
    assert np.allclose(lambdas, scales[:, None] * d[None, :], atol=1e-12)
    assert np.all(penalties == 0.0)


def test_bank_reduces_to_first_order_stakes_for_large_radius() -> None:
    f = np.array([[1.0, 1.4], [1.4, 5.0]])
    scales = np.array([-0.2, 0.2])
    lambdas, _ = robust_bank(f, 50.0, scales, 0.45)
    assert np.allclose(lambdas[:, 1], 0.0, atol=1e-12)
    assert np.allclose(lambdas[:, 0], scales / np.sqrt(f[0, 0]))


def test_width_ratio_endpoints_and_monotonicity() -> None:
    g = 0.6
    rho = np.linspace(0.0, 1.5, 301)
    u = width_ratio(g, rho)
    assert np.isclose(u[0], np.sqrt(g))
    assert np.allclose(u[rho >= 1.0], 1.0)
    assert np.all(np.diff(u) >= -1e-12)


def test_dominating_matrix_covers_region_corners() -> None:
    center, radius, m3c, m3r, m4u = 1.0, 0.05, 1.4, 0.1, 6.5
    fbar = dominating_matrix(center, radius, m3c, m3r, m4u)
    for s in (center - radius, center + radius):
        for m3 in (m3c - m3r, m3c + m3r):
            for delta in (-radius, radius):
                cov = np.array([[s, m3], [m3, m4u - s * s]])
                gap = fbar - cov - np.diag([0.0, delta**2])
                assert np.min(np.linalg.eigvalsh(gap)) >= -1e-12


def test_catoni_bound_coverage() -> None:
    rng = np.random.default_rng(7)
    misses = 0
    reps = 400
    for _ in range(reps):
        y = rng.exponential(size=300)
        misses += catoni_bound(y, 1.0, 0.05, "upper") < 1.0
    # Expected misses <= 20; allow Monte Carlo slack.
    assert misses <= 32


def test_calibration_region_contains_truth_in_a_typical_draw() -> None:
    law = get_law("gamma2")
    rng = np.random.default_rng(11)
    eps = law.sample(rng, 200_000)
    deltas = {"crude_mu4": 0.002, "variance": 0.004, "mu3": 0.002, "mu4": 0.002}
    region = calibrate_moments(eps, 2.0 * 3752.0, deltas, 0.2)
    assert region.covers(1.0, law.mu3, law.mu4)
    assert region.radius < 0.05


def _central(m1: float, m2: float, m3: float, m4: float) -> tuple[float, float, float]:
    return (m2 - m1**2, m3 - 3 * m1 * m2 + 2 * m1**3,
            m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4)


def test_central_from_raw_is_exact_on_point_intervals() -> None:
    atoms, p, mom = _three_point_law()
    raw = [float(p @ (atoms - 0.3) ** k) for k in (1, 2, 3, 4)]
    var, mu3, mu4 = central_from_raw(*[(r, r) for r in raw])
    for got, want in ((var, mom[2]), (mu3, mom[3]), (mu4, mom[4])):
        assert np.isclose(got[0], want) and np.isclose(got[1], want)


def test_central_from_raw_encloses_every_point_of_the_box() -> None:
    rng = np.random.default_rng(5)
    box = [(-0.05, 0.08), (0.9, 1.1), (1.2, 1.6), (5.0, 7.0)]
    var, mu3, mu4 = central_from_raw(*box)
    for _ in range(5000):
        point = [rng.uniform(lo, hi) for lo, hi in box]
        v, m3, m4 = _central(*point)
        assert var[0] - 1e-12 <= v <= var[1] + 1e-12
        assert mu3[0] - 1e-12 <= m3 <= mu3[1] + 1e-12
        assert m4 <= mu4[1] + 1e-12


def test_unknown_center_region_is_location_invariant_and_contains_truth() -> None:
    law = get_law("gamma2")
    eps = law.sample(np.random.default_rng(13), 200_000)
    deltas = {"shift": 0.001, "crude_mu4": 0.001, "mean": 0.002,
              "variance": 0.002, "mu3": 0.002, "mu4": 0.002}
    regions = [calibrate_moments_unknown_center(eps + loc, 2.0 * 3752.0, deltas, 0.02, 0.2)
               for loc in (0.0, 41.5)]
    a, b = regions
    assert np.isclose(a.center, b.center, rtol=1e-6)
    assert np.isclose(a.mu3_radius, b.mu3_radius, rtol=1e-4)
    assert b.covers(1.0, law.mu3, law.mu4)
    linear = {"shift": 0.001, "crude_mu4": 0.001, "variance_upper": 0.008}
    assert calibrate_variance_upper_unknown_center(eps + 41.5, 2.0 * 3752.0, linear,
                                                   0.02, 0.2) >= 1.0
