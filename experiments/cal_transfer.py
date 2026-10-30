"""CAL-1..CAL-3: moments transferred from an independent calibration sample.

Every replication draws its own reference sample of known-zero-mean errors,
builds the moment region and the first-order variance bound from it, and only
then sees the tested stream.  CAL-1 audits validity at the true location,
CAL-3 compares stopping times under shifts, and CAL-2 compares confidence
sequence widths with the matched-boundary prediction.

    PYTHONPATH=src python3 experiments/cal_transfer.py [cal13|cal2|all]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from math import comb
import json
import sys

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from ku_betting_cs.calibration import (
    calibrate_moments,
    calibrate_moments_unknown_center,
    calibrate_variance_upper,
    calibrate_variance_upper_unknown_center,
    robust_bank,
    robust_log_mixture_path,
    robust_terminal_grid,
    width_ratio,
)
from ku_betting_cs.core import (
    crossing_times,
    cs_width_from_grid,
    log_mixture_path,
    terminal_log_mixture_grid,
)
from ku_betting_cs.laws import get_law
from ku_betting_cs.stats import clopper_pearson_upper

from common import ROOT, jsonable


def load_calibration_config(name: str = "calibration_gates.json") -> dict:
    return json.loads((ROOT / "configs" / name).read_text())


def bank(config: dict, two_sided: bool) -> np.ndarray:
    alpha_e = float(config["alpha_e"])
    positive = np.sqrt(2.0 * np.log(2.0 / alpha_e)
                       / np.asarray(config["scale_horizons"], dtype=float))
    return np.concatenate([-positive[::-1], positive]) if two_sided else positive


def eighth_moment(name: str) -> float:
    """Standardized eighth central moment of a registered law."""
    if name in {"gamma2", "lognormal04"}:
        dist = stats.gamma(a=2.0) if name == "gamma2" else stats.lognorm(s=0.4)
        mean, sd = float(dist.mean()), float(dist.std())
        return float(dist.expect(lambda x: ((x - mean) / sd) ** 8))
    nonzero, log_sigma = {"ziln_p25": (0.25, 1.0), "ziln_p60": (0.60, 0.7)}[name]
    raw = [1.0] + [nonzero * np.exp(0.5 * (k * log_sigma) ** 2) for k in range(1, 9)]
    mean = raw[1]
    central = [sum(comb(k, j) * raw[j] * (-mean) ** (k - j) for j in range(k + 1))
               for k in range(9)]
    return float(central[8] / central[2] ** 4)


def calibrate(config: dict, law, eps: np.ndarray, k8: float):
    if config.get("reference") == "unknown_location":
        # Same draws, observed as a record of unknown location (CAL-4).
        y = eps + config["reference_location"]
        args = (config["location_fraction"], config["crude_fraction"])
        region = calibrate_moments_unknown_center(y, k8, config["delta_split_moment"], *args)
        s_lin = calibrate_variance_upper_unknown_center(y, k8, config["delta_split_linear"],
                                                        *args)
        return region, s_lin, region.covers(1.0, law.mu3, law.mu4), bool(s_lin >= 1.0)
    region = calibrate_moments(eps, k8, config["delta_split_moment"],
                               config["crude_fraction"])
    s_lin = calibrate_variance_upper(eps, k8, config["delta_split_linear"],
                                     config["crude_fraction"])
    covered = region.covers(1.0, law.mu3, law.mu4)
    return region, s_lin, covered, bool(s_lin >= 1.0)


# --------------------------------------------------------------- CAL-1 / CAL-3

def _cell13(args: tuple) -> dict:
    config, law_name, n_cal, seed = args
    law = get_law(law_name)
    rng = np.random.default_rng(seed)
    k8 = config["eighth_moment_envelope_factor"] * eighth_moment(law_name)
    g1, g3 = config["CAL-1"], config["CAL-3"]
    log_thr = np.log(1.0 / config["alpha_e"])
    two, one = bank(config, True), bank(config, False)
    cap = config["penalty_cap"]
    crossings = {"robust": 0, "linear": 0}
    region_miss = linear_miss = 0
    stops = {d: {k: [] for k in ("robust", "linear", "oracle_poly", "oracle_lin")}
             for d in g3["signals"]}
    radii, g2bars = [], []
    for _ in range(g1["replications"]):
        eps = law.sample(rng, n_cal)
        region, s_lin, covered, lin_ok = calibrate(config, law, eps, k8)
        region_miss += not covered
        linear_miss += not lin_ok
        radii.append(region.radius)
        g2bars.append(region.g2_bar)
        lam2, pen2 = robust_bank(region.fbar, region.radius, two, cap)
        x = law.sample(rng, g1["horizon"])
        loge = robust_log_mixture_path(x, 0.0, region.center, lam2, pen2, region.fbar)
        crossings["robust"] += bool(np.max(loge) >= log_thr)
        loge = log_mixture_path(x / np.sqrt(s_lin), 0.0, 0.0, 0.0, "linear", two)
        crossings["linear"] += bool(np.max(loge) >= log_thr)
        lam1, pen1 = robust_bank(region.fbar, region.radius, one, cap)
        for delta in g3["signals"]:
            x = law.sample(rng, g3["horizon"]) + delta
            paths = {
                "robust": robust_log_mixture_path(x, 0.0, region.center, lam1, pen1,
                                                  region.fbar),
                "linear": log_mixture_path(x / np.sqrt(s_lin), 0.0, 0.0, 0.0,
                                           "linear", one),
                "oracle_poly": log_mixture_path(x, 0.0, law.mu3, law.mu4,
                                                "polynomial", one),
                "oracle_lin": log_mixture_path(x, 0.0, law.mu3, law.mu4, "linear", one),
            }
            for key, path in paths.items():
                stops[delta][key].append(int(crossing_times(path, config["alpha_e"])))
    reps = g1["replications"]
    null_rows = {
        k: {"crossings": v, "rate": v / reps,
            "cp_upper": clopper_pearson_upper(v, reps, g1["cp_level"])}
        for k, v in crossings.items()
    }
    alt_rows = []
    for delta, by in stops.items():
        arr = {k: np.asarray(v) for k, v in by.items()}
        med = {k: float(np.median(v)) for k, v in arr.items()}
        alt_rows.append({
            "delta": delta,
            "median_stop": med,
            "detection_fraction": {k: float(np.mean(v <= g3["horizon"]))
                                   for k, v in arr.items()},
            "median_gain_robust_vs_linear": 1.0 - med["robust"] / med["linear"],
            "median_gain_oracle_poly_vs_oracle_lin":
                1.0 - med["oracle_poly"] / med["oracle_lin"],
        })
    return {
        "law": law_name,
        "n_cal": n_cal,
        "eighth_moment_envelope": k8,
        "replications": reps,
        "region_misses": region_miss,
        "linear_bound_misses": linear_miss,
        "median_radius": float(np.median(radii)),
        "median_g2_bar": float(np.median(g2bars)),
        "g2": law.g2,
        "null": null_rows,
        "alternatives": alt_rows,
    }


def run_cal13(config: dict) -> bool:
    seeds = np.random.SeedSequence(config["seed"]).spawn(64)
    jobs = []
    for i, law_name in enumerate(config["CAL-1"]["laws"]):
        for j, n_cal in enumerate(config["calibration_sizes"]):
            seed = seeds[3 * i + j].generate_state(1)[0]
            jobs.append((config, law_name, n_cal, int(seed)))
    with ProcessPoolExecutor(max_workers=min(len(jobs), 10)) as pool:
        cells = list(pool.map(_cell13, jobs))
    cp_max = config["CAL-1"]["cp_upper_max"]
    cal1_pass = all(c["null"][k]["cp_upper"] <= cp_max
                    for c in cells for k in ("robust", "linear"))
    g3 = config["CAL-3"]
    largest = max(config["calibration_sizes"])
    gains = [a["median_gain_robust_vs_linear"] for c in cells
             if c["law"] in g3["gain_laws"] and c["n_cal"] == largest
             for a in c["alternatives"]]
    cal3_pass = max(gains) >= g3["required_median_gain_at_largest_calibration"]
    for c in cells:
        print(c["law"], c["n_cal"], "null",
              {k: round(v["rate"], 4) for k, v in c["null"].items()},
              "gains", [round(a["median_gain_robust_vs_linear"], 3)
                        for a in c["alternatives"]], flush=True)
    payload = {"gates": {"CAL-1": cal1_pass, "CAL-3": cal3_pass},
               "cells": cells, "config": {k: config[k] for k in config
                                          if k not in {"CAL-2"}}}
    out = config.get("output_13", "cal_transfer_13.json")
    (ROOT / "results" / out).write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    print("CAL-1", "PASS" if cal1_pass else "FAIL")
    print("CAL-3", "PASS" if cal3_pass else "FAIL")
    return cal1_pass and cal3_pass


# ----------------------------------------------------------------------- CAL-2

def _predicted_halfwidth(w_lin: float, s_lin: float, scales: np.ndarray,
                         lambdas: np.ndarray, penalties: np.ndarray,
                         fbar: np.ndarray, delta_c: float, side: int) -> float:
    """Matched-boundary prediction of one robust half-width.

    The common boundary level is read off the observed first-order
    half-width; local growth of a component is ``lambda.T E(phi) - penalty -
    lambda.T Fbar lambda / 2`` (Lemma 1 without its remainder).
    """
    s = scales[scales * side > 0]
    level = float(np.max(np.abs(s) * w_lin / np.sqrt(s_lin) - 0.5 * s * s))
    v = np.einsum("ki,ij,kj->k", lambdas, fbar, lambdas)

    def growth(w: float) -> float:
        mean = np.stack([np.full_like(v, side * w), np.full_like(v, delta_c + w * w)])
        return float(np.max(np.sum(lambdas.T * mean, axis=0) - penalties - 0.5 * v))

    hi = w_lin
    while growth(hi) < level:
        hi *= 1.5
        if hi > 100.0 * w_lin:
            return float("nan")
    return float(brentq(lambda w: growth(w) - level, 0.0, hi, xtol=1e-12))


def _cell2(args: tuple) -> list[dict]:
    config, law_name, n_cal, seed = args
    gate = config["CAL-2"]
    law = get_law(law_name)
    rng = np.random.default_rng(seed)
    k8 = config["eighth_moment_envelope_factor"] * eighth_moment(law_name)
    alpha_e = config["alpha_e"]
    scales = bank(config, True)
    rows = {n: {"robust": [], "linear": [], "oracle_poly": [], "oracle_lin": [],
                "predicted_ratio": [], "rho_design": [], "closed_form_ratio": [],
                "edge": 0} for n in gate["checkpoints"]}
    for _ in range(gate["replications"]):
        eps = law.sample(rng, n_cal)
        region, s_lin, _, _ = calibrate(config, law, eps, k8)
        lam, pen = robust_bank(region.fbar, region.radius, scales, config["penalty_cap"])
        f = region.fbar
        stream = law.sample(rng, max(gate["checkpoints"]))
        for n in gate["checkpoints"]:
            x = stream[:n]
            radius = gate["grid_se_radius"] * np.sqrt(max(s_lin, f[0, 0])) / np.sqrt(n)
            grid = np.linspace(x.mean() - radius, x.mean() + radius, gate["grid_points"])
            evals = {
                "robust": robust_terminal_grid(x, grid, region.center, lam, pen, f),
                "linear": terminal_log_mixture_grid(x / np.sqrt(s_lin), grid / np.sqrt(s_lin),
                                                    0.0, 0.0, "linear", scales),
                "oracle_poly": terminal_log_mixture_grid(x, grid, law.mu3, law.mu4,
                                                         "polynomial", scales),
                "oracle_lin": terminal_log_mixture_grid(x, grid, law.mu3, law.mu4,
                                                        "linear", scales),
            }
            r = rows[n]
            for key, loge in evals.items():
                width, edge = cs_width_from_grid(grid, loge, alpha_e)
                r[key].append(width)
                r["edge"] += int(edge)
            w_lin = 0.5 * r["linear"][-1]
            delta_c = 1.0 - region.center
            sides = [_predicted_halfwidth(w_lin, s_lin, scales, lam, pen, f, delta_c, sd)
                     for sd in (1, -1)]
            r["predicted_ratio"].append(0.5 * sum(sides) / w_lin)
            s_eff = w_lin / np.sqrt(s_lin)
            rho = region.radius * np.sqrt(f[0, 0]) / (abs(f[0, 1]) * s_eff)
            r["rho_design"].append(rho)
            r["closed_form_ratio"].append(
                float(width_ratio(region.g2_bar, rho)) * np.sqrt(f[0, 0] / s_lin))
    out = []
    for n, r in rows.items():
        ratio = np.asarray(r["robust"]) / np.asarray(r["linear"])
        out.append({
            "law": law_name, "n_cal": n_cal, "n": n, "g2": law.g2,
            "mean_ratio_robust_linear": float(ratio.mean()),
            "mean_predicted_ratio": float(np.nanmean(r["predicted_ratio"])),
            "mean_closed_form_ratio": float(np.mean(r["closed_form_ratio"])),
            "median_rho_design": float(np.median(r["rho_design"])),
            "mean_ratio_oracle": float(np.mean(np.asarray(r["oracle_poly"])
                                               / np.asarray(r["oracle_lin"]))),
            "mean_width": {k: float(np.mean(r[k])) for k in
                           ("robust", "linear", "oracle_poly", "oracle_lin")},
            "per_path_ratio": ratio.tolist(),
            "per_path_rho_design": r["rho_design"],
            "edge_touches": r["edge"],
        })
    return out


def run_cal2(config: dict) -> bool:
    gate = config["CAL-2"]
    seeds = np.random.SeedSequence(config["seed"] + 2).spawn(64)
    jobs = []
    for i, law_name in enumerate(gate["laws"]):
        for j, n_cal in enumerate(config["calibration_sizes"]):
            jobs.append((config, law_name, n_cal,
                         int(seeds[3 * i + j].generate_state(1)[0])))
    with ProcessPoolExecutor(max_workers=min(len(jobs), 10)) as pool:
        rows = [row for cell in pool.map(_cell2, jobs) for row in cell]
    passed = True
    for row in rows:
        ok = (abs(row["mean_ratio_robust_linear"] - row["mean_predicted_ratio"])
              <= gate["relative_tolerance"] * row["mean_predicted_ratio"]
              and row["edge_touches"] == 0)
        row["pass"] = bool(ok)
        if row["n"] >= 1000:
            passed &= ok
        print(row["law"], row["n_cal"], row["n"],
              f"emp={row['mean_ratio_robust_linear']:.3f}",
              f"pred={row['mean_predicted_ratio']:.3f}",
              f"closed={row['mean_closed_form_ratio']:.3f}",
              f"oracle={row['mean_ratio_oracle']:.3f}", ok, flush=True)
    payload = {"gate": "CAL-2", "pass": passed, "rows": rows, "config": gate}
    (ROOT / "results" / "cal_transfer_2.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    print("CAL-2", "PASS" if passed else "FAIL")
    return passed


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "cal4":
        # CAL-4 reuses the CAL-1/CAL-3 cells with a reference of unknown location.
        return 0 if run_cal13(load_calibration_config("calibration_unknown_center.json")) else 1
    config = load_calibration_config()
    ok = True
    if what in {"cal13", "all"}:
        ok &= run_cal13(config)
    if what in {"cal2", "all"}:
        ok &= run_cal2(config)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
