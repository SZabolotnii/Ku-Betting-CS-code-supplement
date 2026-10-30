"""BET-5: sequential sample-to-decision on revenue-like laws."""

from __future__ import annotations

import numpy as np

from ku_betting_cs.core import crossing_times, log_mixture_path
from ku_betting_cs.laws import get_law

from common import jsonable, load_config, scale_bank, write_json


METHODS = ("polynomial", "linear", "catoni")
BOOTSTRAP_REPS = 2000


def median_gain_ci(poly: np.ndarray, linear: np.ndarray,
                   rng: np.random.Generator) -> tuple[float, float]:
    """Paired bootstrap interval for 1-median(poly)/median(linear)."""
    n = len(poly)
    values = np.empty(BOOTSTRAP_REPS)
    for start in range(0, BOOTSTRAP_REPS, 100):
        stop = min(start + 100, BOOTSTRAP_REPS)
        idx = rng.integers(0, n, size=(stop - start, n))
        values[start:stop] = 1.0 - np.median(poly[idx], axis=1) / np.median(
            linear[idx], axis=1
        )
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def main() -> int:
    config = load_config()
    gate = config["BET-5"]
    rng = np.random.default_rng(config["seed"] + 5)
    bootstrap_rng = np.random.default_rng(config["seed"] + 5005)
    scales = np.abs(scale_bank(config, two_sided=False))
    rows = []
    gains = []
    for law_name in gate["laws"]:
        law = get_law(law_name)
        for delta in gate["signals"]:
            stops = {method: [] for method in METHODS}
            for start in range(0, gate["replications"], gate["batch"]):
                batch = min(gate["batch"], gate["replications"] - start)
                x = law.sample(rng, (batch, gate["horizon"])) + delta
                for method in METHODS:
                    loge = log_mixture_path(x, 0.0, law.mu3, law.mu4, method, scales)
                    stops[method].append(crossing_times(loge, config["alpha"]))
            stops = {k: np.concatenate(v) for k, v in stops.items()}
            med = {k: float(np.median(v)) for k, v in stops.items()}
            gain = 1.0 - med["polynomial"] / med["linear"]
            gain_ci = median_gain_ci(
                stops["polynomial"], stops["linear"], bootstrap_rng
            )
            iqr = {
                k: [float(q) for q in np.quantile(v, [0.25, 0.75])]
                for k, v in stops.items()
            }
            gains.append(gain)
            detected = {k: float(np.mean(v <= gate["horizon"])) for k, v in stops.items()}
            rows.append(
                {
                    "law": law_name,
                    "delta": delta,
                    "g2": law.g2,
                    "median_stop": med,
                    "iqr_stop": iqr,
                    "detection_fraction": detected,
                    "median_gain_vs_linear": gain,
                    "bootstrap95_gain": gain_ci,
                    "bootstrap_replications": BOOTSTRAP_REPS,
                    "gain_at_least_15pct": gain >= gate["required_median_gain"],
                }
            )
            print(law_name, delta, med, f"gain={gain:.3f}")
    # The registered claim requires a gain in at least one realistic scenario,
    # while all no-gain cases remain in the result table.
    passed = max(gains) >= gate["required_median_gain"]
    payload = {"gate": "BET-5", "pass": passed, "rows": rows, "config": gate}
    write_json("bet5_ab.json", jsonable(payload))
    print("BET-5", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
