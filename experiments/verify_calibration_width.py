"""Check the calibrated local-width law symbolically and by brute force.

Two claims of the calibrated-moment proposition are checked:

1. the Schur identity
   ``(w, y) F^{-1} (w, y)^T = w^2/f11 + (y - f12 w/f11)^2 / (f22 - f12^2/f11)``
   and the root ``u(rho)`` of the resulting width equation (sympy);
2. the minimax statement: the best first-order growth over directions,
   ``max_a min_{|y|<=r} a.(w, y) / sqrt(a F a)``, equals the F^{-1}-distance
   from the origin to ``{(w, y): |y| <= r}``, so the half-width ratio equals
   ``u(rho)`` (dense search over directions on random instances).
"""

from __future__ import annotations

import json

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from ku_betting_cs.calibration import width_ratio

from common import ROOT


def symbolic() -> bool:
    w, y, f11, f22, r, rho, g, u = sp.symbols("w y f11 f22 r rho g u", positive=True)
    f12 = sp.symbols("f12", real=True)
    F = sp.Matrix([[f11, f12], [f12, f22]])
    v = sp.Matrix([w, y])
    lhs = (v.T * F.inv() * v)[0]
    rhs = w**2 / f11 + (y - f12 * w / f11) ** 2 / (f22 - f12**2 / f11)
    schur_ok = sp.simplify(lhs - rhs) == 0
    # Width equation in standardized form: g u^2 + (1-g)(u-rho)^2 = g.
    closed = (1 - g) * rho + sp.sqrt(g * (1 - (1 - g) * rho**2))
    root_ok = sp.simplify(sp.expand(g * closed**2 + (1 - g) * (closed - rho) ** 2 - g)) == 0
    ends_ok = (sp.simplify(closed.subs(rho, 0) - sp.sqrt(g)) == 0
               and sp.simplify(closed.subs(rho, 1) - 1) == 0)
    return bool(schur_ok and root_ok and ends_ok)


def brute_force(instances: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, 20001)
    dirs = np.stack([np.cos(theta), np.sin(theta)])
    worst = 0.0
    for _ in range(instances):
        f11, f22 = rng.uniform(0.5, 2.0), rng.uniform(1.0, 10.0)
        f12 = rng.uniform(-0.95, 0.95) * np.sqrt(f11 * f22)
        F = np.array([[f11, f12], [f12, f22]])
        level = rng.uniform(1e-3, 1.0)
        radius = rng.uniform(0.0, 1.5) * abs(f12) * level / np.sqrt(f11)
        den = np.sqrt(np.einsum("ik,ij,jk->k", dirs, F, dirs))

        def rate(w: float) -> float:
            num = dirs[0] * w - np.abs(dirs[1]) * radius
            return float(np.max(np.maximum(num, 0.0) / den) ** 2)

        w_best = brentq(lambda w: rate(w) - level**2, 1e-12, 10.0 * level * np.sqrt(f11))
        g = 1.0 - f12**2 / (f11 * f22)
        rho = radius * np.sqrt(f11) / (abs(f12) * level)
        predicted = float(width_ratio(g, rho))
        worst = max(worst, abs(w_best / (level * np.sqrt(f11)) - predicted))
    return worst


def main() -> int:
    sym = symbolic()
    worst = brute_force(2000, 20261030)
    passed = bool(sym and worst < 1e-5)
    payload = {"symbolic_identities": sym, "brute_force_instances": 2000,
               "max_abs_error": worst, "tolerance": 1e-5, "pass": passed}
    (ROOT / "results" / "calibration_width_check.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print("CAL-WIDTH-LAW", "PASS" if passed else "FAIL", payload)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
