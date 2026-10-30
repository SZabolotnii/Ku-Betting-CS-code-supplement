"""BET-4: signal sweep for the quadratic GROW approximation."""

from __future__ import annotations

import numpy as np

from ku_betting_cs.core import moment_matrix
from ku_betting_cs.laws import get_law

from common import jsonable, load_config, write_json


def main() -> int:
    config = load_config()
    gate = config["BET-4"]
    rng = np.random.default_rng(config["seed"] + 4)
    rows = []
    passed = True
    for law_name in gate["laws"]:
        law = get_law(law_name)
        base = law.sample(rng, gate["mc_size"])
        f = moment_matrix(law.mu3, law.mu4)
        for delta in gate["signals"]:
            beta = np.array([delta, delta**2])
            lam = np.linalg.solve(f, beta)
            v = float(lam @ f @ lam)
            x = base + delta
            phi = np.column_stack([x, x * x - 1.0])
            h = phi @ lam
            # Control variate: E_Q h = lambda^T beta = v exactly.  Estimating
            # only log q(h)-h removes the O(delta) Monte Carlo noise while
            # preserving the exact target.
            exact = float(
                v
                + np.mean(np.log1p(h + h * h / 2.0) - h)
                - np.log1p(v / 2.0)
            )
            quadratic = v / 2.0
            rel = abs(exact - quadratic) / max(quadratic, 1e-300)
            third = float(np.mean(np.abs(h) ** 3)) / 3.0
            bound = third + v * v / 8.0
            bound_ok = abs(exact - quadratic) <= bound + 5e-4
            local_ok = rel <= gate["local_relative_error_max"] if delta <= 0.1 else True
            passed &= bound_ok and local_ok
            rows.append(
                {
                    "law": law_name,
                    "delta": delta,
                    "exact_growth": exact,
                    "quadratic_growth": quadratic,
                    "relative_error": rel,
                    "remainder_bound": bound,
                    "bound_ok": bound_ok,
                    "local_ok": local_ok,
                }
            )
            print(law_name, delta, f"rel={rel:.3f}", f"bound={bound:.4g}", bound_ok, local_ok)
    for law_name in gate["laws"]:
        subset = [r for r in rows if r["law"] == law_name]
        bad = [r["delta"] for r in subset if r["relative_error"] > gate["local_relative_error_max"]]
        crossover = min(bad) if bad else None
        for row in subset:
            row["ten_percent_crossover"] = crossover
    payload = {"gate": "BET-4", "pass": passed, "rows": rows, "config": gate}
    write_json("bet4_growth.json", jsonable(payload))
    print("BET-4", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
