import numpy as np

from ku_betting_cs.core import (
    catoni_influence,
    log_factor,
    log_quadratic_factor,
    moment_matrix,
    predictable_ftrl_path,
    score_direction,
)
from ku_betting_cs.laws import get_law


def test_quadratic_factor_is_positive() -> None:
    z = np.array([-1e8, -10.0, -1.0, 0.0, 3.0, 1e8])
    factors = np.exp(log_factor(z, np.array([0.3]), "linear")[:, 0])
    assert np.all(factors > 0.0)


def test_factor_expectation_is_one_with_exact_second_moment() -> None:
    rng = np.random.default_rng(2026)
    x = rng.normal(size=2_000_000)
    factor = np.exp(log_factor(x, np.array([0.2]), "linear")[:, 0])
    assert abs(factor.mean() - 1.0) < 2e-3


def test_polynomial_scores_have_unit_null_variance() -> None:
    for name in ["gaussian", "gamma2", "lognormal04", "skew_mixture", "pareto5"]:
        law = get_law(name)
        f = moment_matrix(law.mu3, law.mu4)
        d = score_direction(law.mu3, law.mu4, "polynomial")
        assert np.isclose(d @ f @ d, 1.0, atol=1e-12)


def test_g2_matches_inverse_information() -> None:
    for name in ["gamma2", "lognormal04", "skew_mixture"]:
        law = get_law(name)
        f = moment_matrix(law.mu3, law.mu4)
        information = np.linalg.inv(f)[0, 0]
        assert np.isclose(information, 1.0 / law.g2, atol=1e-12)


def test_catoni_influence_is_odd() -> None:
    x = np.linspace(-5, 5, 1001)
    assert np.allclose(catoni_influence(-x), -catoni_influence(x))


def test_growth_remainder_bound() -> None:
    u = np.linspace(-8.0, 8.0, 100001)
    qlog = np.log1p(u + u * u / 2.0)
    assert np.all(np.abs(qlog - u) <= np.abs(u) ** 3 / 3.0 + 1e-12)
    v = np.linspace(0.0, 20.0, 100001)
    r = v / 2.0 - np.log1p(v / 2.0)
    assert np.all((r >= -1e-14) & (r <= v * v / 8.0 + 1e-12))


def test_general_quadratic_factor_matches_formula() -> None:
    h = np.array([-3.0, -0.2, 0.0, 1.7])
    v = np.array([0.1, 0.2, 0.3, 0.4])
    expected = np.log(1.0 + h + h * h / 2.0) - np.log(1.0 + v / 2.0)
    assert np.allclose(log_quadratic_factor(h, v), expected)


def test_ftrl_stakes_are_strictly_predictable() -> None:
    z = np.array([0.4, -0.2, 1.5, -0.7, 0.1])
    _, stakes = predictable_ftrl_path(z, regularization=10.0, cap=0.3)
    assert stakes[0] == 0.0
    for t in range(len(z)):
        changed = z.copy()
        changed[t] += 1000.0
        _, changed_stakes = predictable_ftrl_path(
            changed, regularization=10.0, cap=0.3
        )
        assert changed_stakes[t] == stakes[t]


def test_ftrl_formula_uses_only_past_prefix() -> None:
    z = np.array([1.0, 2.0, -1.0, 4.0])
    loge, stakes = predictable_ftrl_path(z, regularization=2.0, cap=10.0)
    expected = np.array([0.0, 1.0 / 3.0, 3.0 / 4.0, 2.0 / 5.0])
    assert np.allclose(stakes, expected)
    increments = log_quadratic_factor(stakes * z, stakes**2)
    assert np.allclose(loge, np.cumsum(increments))
