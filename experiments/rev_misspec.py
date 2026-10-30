"""Supplementary finite-horizon audit of oracle-moment misspecification.

This experiment was frozen in ``configs/revision_gates.json`` after the main
study; it is not one of the original preregistered BET gates.  Analytic diagnostics
separate score-centering error from failure of the assumed PSD matrix bound.
"""

from __future__ import annotations

import numpy as np

from ku_betting_cs.core import crossing_times, log_quadratic_factor, moment_matrix
from ku_betting_cs.laws import get_law
from ku_betting_cs.stats import clopper_pearson_upper

from revision_common import load_revision_config, revision_scale_bank, write_revision_json


def assumed_spec(law, scenario: dict) -> tuple[float, float, float, np.ndarray]:
    variance, mu3, mu4 = 1.0, law.mu3, law.mu4
    if scenario["kind"] == "parameter_error":
        error = scenario["error"]
        if scenario["parameter"] == "variance":
            variance *= 1.0 + error
        elif scenario["parameter"] == "mu3":
            if abs(mu3) > 1e-14:
                mu3 *= 1.0 + error
            else:
                mu3 = error * np.sqrt(mu4 - 1.0)
        elif scenario["parameter"] == "mu4":
            mu4 *= 1.0 + error
    f_assumed = moment_matrix(mu3, mu4, variance)
    if scenario["kind"] == "matrix_inflation":
        f_assumed = scenario["factor"] * moment_matrix(law.mu3, law.mu4)
    return variance, mu3, mu4, f_assumed


def scenarios(gate: dict) -> list[dict]:
    rows = [{"id": "oracle", "kind": "oracle"}]
    for parameter in gate["parameters"]:
        for error in gate["relative_errors"]:
            rows.append(
                {
                    "id": f"{parameter}_{error:+.0%}",
                    "kind": "parameter_error",
                    "parameter": parameter,
                    "error": error,
                }
            )
    for factor in gate["matrix_inflation"]:
        rows.append(
            {
                "id": f"matrix_inflation_{factor:.2f}",
                "kind": "matrix_inflation",
                "factor": factor,
            }
        )
    return rows


def prepare(law, scenario: dict, scales: np.ndarray) -> dict:
    variance, mu3, mu4, f_assumed = assumed_spec(law, scenario)
    eig = np.linalg.eigvalsh(f_assumed)
    if eig[0] <= 0.0:
        raise ValueError(f"non-PSD assumed matrix in {law.name}/{scenario['id']}: {eig}")
    raw = np.linalg.solve(f_assumed, np.array([1.0, 0.0]))
    direction = raw / np.sqrt(raw @ f_assumed @ raw)
    b = np.array([0.0, 1.0 - variance])
    s_true = np.array(
        [
            [1.0, law.mu3],
            [law.mu3, law.mu4 - 2.0 * variance + variance**2],
        ]
    )
    centering = float(direction @ b)
    true_second = float(direction @ s_true @ direction)
    psd_error = float(np.linalg.eigvalsh(s_true - f_assumed)[-1])
    expected_factors = (
        1.0 + scales * centering + 0.5 * scales**2 * true_second
    ) / (1.0 + 0.5 * scales**2)
    return {
        "scenario": scenario,
        "variance": variance,
        "mu3": mu3,
        "mu4": mu4,
        "f_assumed": f_assumed,
        "direction": direction,
        "centering_bias_vector": b,
        "directional_centering_bias": centering,
        "true_directional_second_moment": true_second,
        "max_eigenvalue_S_minus_F": psd_error,
        "psd_bound_holds": psd_error <= 1e-12,
        "max_one_step_expected_factor": float(np.max(expected_factors)),
    }


def score_with_spec(x: np.ndarray, prepared: dict) -> np.ndarray:
    d = prepared["direction"]
    return d[0] * x + d[1] * (x * x - prepared["variance"])


def mixture_path(z: np.ndarray, scales: np.ndarray) -> np.ndarray:
    increments = log_quadratic_factor(
        z[..., None] * scales,
        np.broadcast_to(scales**2, z.shape + (len(scales),)),
    )
    wealth = np.cumsum(increments, axis=-2)
    maximum = np.max(wealth, axis=-1, keepdims=True)
    return maximum[..., 0] + np.log(
        np.mean(np.exp(wealth - maximum), axis=-1)
    )


def main() -> int:
    config = load_revision_config()
    gate = config["REV-MISSPEC"]
    scales = revision_scale_bank(config)
    threshold = np.log(1.0 / config["alpha"])
    rng = np.random.default_rng(config["seed"] + 1)
    output = []
    for law_name in gate["laws"]:
        law = get_law(law_name)
        prepared = [prepare(law, s, scales) for s in scenarios(gate)]
        hits = np.zeros(len(prepared), dtype=int)
        stops = [[] for _ in prepared]
        for start in range(0, gate["replications"], gate["batch"]):
            batch = min(gate["batch"], gate["replications"] - start)
            base = law.sample(rng, (batch, gate["horizon"]))
            for index, item in enumerate(prepared):
                null_loge = mixture_path(score_with_spec(base, item), scales)
                hits[index] += int(np.sum(np.max(null_loge, axis=1) >= threshold))
                alt_loge = mixture_path(
                    score_with_spec(base + gate["alternative_shift"], item), scales
                )
                stops[index].append(crossing_times(alt_loge, config["alpha"]))
        for index, item in enumerate(prepared):
            stop = np.concatenate(stops[index])
            rate = hits[index] / gate["replications"]
            output.append(
                {
                    "law": law_name,
                    "scenario": item["scenario"],
                    "centering_bias_vector": item["centering_bias_vector"].tolist(),
                    "directional_centering_bias": item["directional_centering_bias"],
                    "true_directional_second_moment": item["true_directional_second_moment"],
                    "max_eigenvalue_S_minus_F": item["max_eigenvalue_S_minus_F"],
                    "psd_bound_holds": item["psd_bound_holds"],
                    "max_one_step_expected_factor": item["max_one_step_expected_factor"],
                    "null_hits": int(hits[index]),
                    "null_ever_crossing_rate": rate,
                    "null_cp95_upper": clopper_pearson_upper(
                        int(hits[index]), gate["replications"], gate["cp_level"]
                    ),
                    "alternative_shift": gate["alternative_shift"],
                    "median_stopping_time": float(np.median(stop)),
                    "detection_fraction": float(np.mean(stop <= gate["horizon"])),
                }
            )
        print("REV-MISSPEC", law_name, "done")
    payload = {
        "gate": "REV-MISSPEC",
        "status": "supplementary, specified after the main study; not originally preregistered",
        "pass": True,
        "rows": output,
        "config": gate,
    }
    write_revision_json("rev_misspec.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
