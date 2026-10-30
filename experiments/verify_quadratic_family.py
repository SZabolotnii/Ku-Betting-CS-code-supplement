"""Symbolic and numerical audit of local equivalence across quadratic envelopes."""

from __future__ import annotations

import numpy as np
import sympy as sp
from numpy.polynomial.hermite import hermgauss

from revision_common import load_revision_config, write_revision_json


def symbolic_audit() -> dict:
    c, u, v = sp.symbols("c u v", real=True)
    numerator = sp.series(sp.log(1 + u + c * u**2), u, 0, 3).removeO()
    denominator = sp.series(sp.log(1 + c * v), v, 0, 2).removeO()
    numerator_expected = u + (c - sp.Rational(1, 2)) * u**2
    penalty_identity = sp.simplify((c - sp.Rational(1, 2)) * v - denominator)
    return {
        "log_numerator_series": str(sp.expand(numerator)),
        "numerator_series_matches": sp.simplify(numerator - numerator_expected) == 0,
        "second_order_penalty_after_Eu2_eq_v": str(penalty_identity),
        "penalty_is_minus_half_v": sp.simplify(
            penalty_identity + sp.Rational(1, 2) * v
        ) == 0,
    }


def numerical_audit(config: dict) -> list[dict]:
    gate = config["QUADRATIC-FAMILY"]
    nodes, weights = hermgauss(gate["gauss_hermite_nodes"])
    weights = weights / np.sqrt(np.pi)
    rows = []
    rho = gate["local_slope"]
    target = rho - 0.5
    for c in gate["c_values"]:
        for eps in gate["eps_values"]:
            x = np.sqrt(2.0) * nodes + rho * eps
            h = eps * x
            numerator = 1.0 + h + c * h * h
            growth = float(np.sum(weights * np.log(numerator)) - np.log1p(c * eps**2))
            scaled = growth / eps**2
            rows.append(
                {
                    "c": c,
                    "eps": eps,
                    "growth": growth,
                    "growth_over_eps2": scaled,
                    "target": target,
                    "absolute_scaled_error": abs(scaled - target),
                }
            )
    return rows


def main() -> int:
    config = load_revision_config()
    symbolic = symbolic_audit()
    rows = numerical_audit(config)
    smallest = min(config["QUADRATIC-FAMILY"]["eps_values"])
    max_error = max(r["absolute_scaled_error"] for r in rows if r["eps"] == smallest)
    passed = (
        symbolic["numerator_series_matches"]
        and symbolic["penalty_is_minus_half_v"]
        and max_error <= config["QUADRATIC-FAMILY"]["scaled_error_max_at_smallest_eps"]
    )
    payload = {
        "gate": "QUADRATIC-FAMILY",
        "pass": passed,
        "symbolic": symbolic,
        "rows": rows,
        "max_scaled_error_at_smallest_eps": max_error,
        "config": config["QUADRATIC-FAMILY"],
    }
    write_revision_json("quadratic_family.json", payload)
    print("QUADRATIC-FAMILY", "PASS" if passed else "FAIL", max_error)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
