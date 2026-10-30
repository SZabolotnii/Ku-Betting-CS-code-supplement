"""BET-2: paired finite-sample widths of inverted mixture e-processes."""

from __future__ import annotations

import numpy as np

from ku_betting_cs.core import cs_width_from_grid, terminal_log_mixture_grid
from ku_betting_cs.laws import get_law
from ku_betting_cs.stats import paired_bootstrap_ci

from common import jsonable, load_config, scale_bank, write_json


METHODS = ("polynomial", "linear", "catoni")


def main() -> int:
    config = load_config()
    gate = config["BET-2"]
    rng = np.random.default_rng(config["seed"] + 2)
    scales = scale_bank(config, two_sided=True)
    rows = []
    passed = True
    for law_name in gate["laws"]:
        law = get_law(law_name)
        samples = law.sample(rng, (gate["replications"], max(gate["checkpoints"])))
        for n in gate["checkpoints"]:
            widths = {method: [] for method in METHODS}
            touched = {method: 0 for method in METHODS}
            for x in samples[:, :n]:
                center = float(np.mean(x))
                radius = gate["grid_se_radius"] / np.sqrt(n)
                candidates = np.linspace(
                    center - radius, center + radius, gate["grid_points"]
                )
                for method in METHODS:
                    loge = terminal_log_mixture_grid(
                        x, candidates, law.mu3, law.mu4, method, scales
                    )
                    width, edge = cs_width_from_grid(candidates, loge, config["alpha"])
                    widths[method].append(width)
                    touched[method] += int(edge)
            poly = np.asarray(widths["polynomial"])
            linear = np.asarray(widths["linear"])
            catoni = np.asarray(widths["catoni"])
            ratio = poly / linear
            ratio_ci = paired_bootstrap_ci(ratio, rng)
            predicted = np.sqrt(law.g2)
            ratio_ok = (
                abs(float(np.mean(ratio)) - predicted)
                <= gate["asymmetric_ratio_tolerance"] * predicted
                if abs(law.mu3) > 1e-12 and n >= 1000
                else float(np.mean(ratio)) <= 1.0 + gate["gaussian_degradation_max"]
            )
            edge_ok = all(v == 0 for v in touched.values())
            row_pass = ratio_ok and edge_ok
            if n >= 1000:
                passed &= row_pass
            row = {
                "law": law_name,
                "n": n,
                "g2": law.g2,
                "predicted_ratio": predicted,
                "mean_width_polynomial": float(poly.mean()),
                "mean_width_linear": float(linear.mean()),
                "mean_width_catoni": float(catoni.mean()),
                "mean_ratio_poly_linear": float(ratio.mean()),
                "bootstrap95_ratio": ratio_ci,
                "edge_touches": touched,
                "pass": row_pass,
            }
            rows.append(row)
            print(
                law_name,
                n,
                f"ratio={ratio.mean():.3f}",
                f"sqrt(g2)={predicted:.3f}",
                f"catoni={catoni.mean():.4f}",
                row_pass,
            )
    payload = {"gate": "BET-2", "pass": passed, "rows": rows, "config": gate}
    write_json("bet2_width.json", jsonable(payload))
    print("BET-2", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
