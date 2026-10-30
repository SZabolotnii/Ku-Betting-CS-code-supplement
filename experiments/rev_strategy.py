"""Supplementary comparison of fixed and strictly predictable adaptive strategies."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ku_betting_cs.core import (
    crossing_times,
    log_mixture_path,
    predictable_ftrl_path,
    score,
    terminal_log_mixture_grid,
)
from ku_betting_cs.laws import get_law
from ku_betting_cs.stats import clopper_pearson_upper

from revision_common import load_revision_config, revision_scale_bank, write_revision_json


def methods(config: dict) -> list[dict]:
    gate = config["REV-STRATEGY"]
    rows = [
        {"id": "fixed_10", "kind": "fixed", "scales": revision_scale_bank(config)},
        {
            "id": f"dense_{2 * gate['dense_bank_size_per_sign']}",
            "kind": "fixed",
            "scales": revision_scale_bank(config, gate["dense_bank_size_per_sign"]),
        },
    ]
    rows.extend(
        {
            "id": f"ftrl_reg_{regularization:g}",
            "kind": "ftrl",
            "regularization": regularization,
            "cap": gate["ftrl_cap"],
        }
        for regularization in gate["ftrl_regularization"]
    )
    return rows


def method_path(x: np.ndarray, law, item: dict) -> np.ndarray:
    if item["kind"] == "fixed":
        return log_mixture_path(
            x, 0.0, law.mu3, law.mu4, "polynomial", item["scales"]
        )
    z = score(x, 0.0, law.mu3, law.mu4, "polynomial")
    return predictable_ftrl_path(z, item["regularization"], item["cap"])[0]


def null_rows(config: dict, method_specs: list[dict], rng) -> list[dict]:
    gate = config["REV-STRATEGY"]
    threshold = np.log(1.0 / config["alpha"])
    rows = []
    for law_name in gate["null_laws"]:
        law = get_law(law_name)
        hits = np.zeros(len(method_specs), dtype=int)
        for start in range(0, gate["replications"], gate["batch"]):
            batch = min(gate["batch"], gate["replications"] - start)
            x = law.sample(rng, (batch, gate["horizon"]))
            for i, item in enumerate(method_specs):
                loge = method_path(x, law, item)
                hits[i] += int(np.sum(np.max(loge, axis=1) >= threshold))
        for i, item in enumerate(method_specs):
            rows.append(
                {
                    "law": law_name,
                    "method": item["id"],
                    "hits": int(hits[i]),
                    "replications": gate["replications"],
                    "ever_crossing_rate": hits[i] / gate["replications"],
                    "cp95_upper": clopper_pearson_upper(
                        int(hits[i]), gate["replications"]
                    ),
                }
            )
        print("REV-STRATEGY null", law_name, "done")
    return rows


def alternative_rows(config: dict, method_specs: list[dict], rng) -> list[dict]:
    gate = config["REV-STRATEGY"]
    rows = []
    for law_name in gate["alternative_laws"]:
        law = get_law(law_name)
        for delta in gate["signals"]:
            stops = [[] for _ in method_specs]
            for start in range(0, gate["replications"], gate["batch"]):
                batch = min(gate["batch"], gate["replications"] - start)
                x = law.sample(rng, (batch, gate["horizon"])) + delta
                for i, item in enumerate(method_specs):
                    loge = method_path(x, law, item)
                    stops[i].append(crossing_times(loge, config["alpha"]))
            for i, item in enumerate(method_specs):
                stop = np.concatenate(stops[i])
                rows.append(
                    {
                        "law": law_name,
                        "delta": delta,
                        "method": item["id"],
                        "median_stopping_time": float(np.median(stop)),
                        "iqr_stopping_time": np.quantile(stop, [0.25, 0.75]).tolist(),
                        "detection_fraction": float(np.mean(stop <= gate["horizon"])),
                    }
                )
            print("REV-STRATEGY alternative", law_name, delta, "done")
    return rows


def timing_rows(config: dict, method_specs: list[dict], rng) -> list[dict]:
    gate = config["REV-STRATEGY"]
    law = get_law("gamma2")
    paths = law.sample(rng, (gate["timing_paths"], gate["timing_horizon"]))
    rows = []
    radius = 8.0 / np.sqrt(gate["timing_horizon"])
    for item in method_specs:
        started = perf_counter()
        evaluations = 0
        for x in paths:
            candidates = np.linspace(x.mean() - radius, x.mean() + radius, gate["timing_grid_points"])
            if item["kind"] == "fixed":
                terminal_log_mixture_grid(
                    x,
                    candidates,
                    law.mu3,
                    law.mu4,
                    "polynomial",
                    item["scales"],
                )
            else:
                for candidate in candidates:
                    z = score(x, candidate, law.mu3, law.mu4, "polynomial")
                    predictable_ftrl_path(z, item["regularization"], item["cap"])
            evaluations += len(candidates)
        elapsed = perf_counter() - started
        rows.append(
            {
                "method": item["id"],
                "law": law.name,
                "paths": gate["timing_paths"],
                "horizon": gate["timing_horizon"],
                "grid_points": gate["timing_grid_points"],
                "elapsed_seconds": elapsed,
                "milliseconds_per_candidate": 1000.0 * elapsed / evaluations,
                "machine_dependent": True,
            }
        )
    return rows


def main() -> int:
    config = load_revision_config()
    specs = methods(config)
    rng = np.random.default_rng(config["seed"] + 2)
    payload = {
        "gate": "REV-STRATEGY",
        "status": "supplementary, specified after the main study; not originally preregistered",
        "pass": True,
        "predictability": "lambda_t uses only scores through t-1; unit-tested",
        "null": null_rows(config, specs, rng),
        "alternatives": alternative_rows(config, specs, rng),
        "inversion_timing": timing_rows(config, specs, rng),
        "config": config["REV-STRATEGY"],
    }
    write_revision_json("rev_strategy.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
