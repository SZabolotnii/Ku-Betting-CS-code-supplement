"""BET-3: exact 1-kappa identity on finite QMD anchors."""

from __future__ import annotations

import numpy as np

from common import jsonable, load_config, write_json


def anchor(name: str, x: np.ndarray, p: np.ndarray, theta: float,
           tangent: np.ndarray) -> dict:
    q_unnorm = p * np.exp(theta * tangent)
    q = q_unnorm / q_unnorm.sum()
    z = q / p - 1.0
    phi = np.column_stack([x - p @ x, x**2 - p @ (x**2)])
    f = phi.T @ (p[:, None] * phi)
    beta = q @ phi
    j = float(beta @ np.linalg.solve(f, beta))
    chi2 = float(p @ (z * z))
    kappa = j / chi2
    deficit_direct = (chi2 - j) / 2.0
    deficit_price = (1.0 - kappa) * chi2 / 2.0
    return {
        "anchor": name,
        "theta": theta,
        "chi2": chi2,
        "J": j,
        "kappa": kappa,
        "deficit_direct": deficit_direct,
        "deficit_price": deficit_price,
        "relative_error": abs(deficit_direct - deficit_price) / max(chi2, 1e-300),
    }


def main() -> int:
    config = load_config()
    gate = config["BET-3"]
    rows = [
        anchor("cubic-5-point", np.arange(-2, 3), np.array([1, 4, 6, 4, 1]) / 16, 0.18, np.arange(-2, 3) ** 3),
        anchor("threshold-5-point", np.array([-2, -1, 0, 1, 3.0]), np.array([.08, .17, .31, .29, .15]), 0.55, np.array([0, 0, 0, 1, 1.0])),
        anchor("sinusoidal-6-point", np.array([-3, -1.5, -.5, .5, 1.5, 2.5]), np.array([.06, .14, .25, .27, .19, .09]), -0.8, np.sin(np.array([-3, -1.5, -.5, .5, 1.5, 2.5]))),
    ]
    passed = len(rows) >= gate["minimum_anchors"] and all(
        r["relative_error"] <= gate["relative_tolerance"] for r in rows
    )
    write_json("bet3_kappa.json", jsonable({"gate": "BET-3", "pass": passed, "rows": rows, "config": gate}))
    for row in rows:
        print(row["anchor"], f"kappa={row['kappa']:.6f}", f"error={row['relative_error']:.2e}")
    print("BET-3", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
