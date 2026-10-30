"""Largest penalty |lambda_2| r over the CAL-1/CAL-3 replications.

Replays the random-number consumption of ``cal_transfer._cell13`` exactly
(reference sample, null stream, one stream per shift) so that the audited
calibrations are the ones used in the formal run, and records whether any
stake reached the positivity cap.
"""

from __future__ import annotations

import json

import numpy as np

from ku_betting_cs.calibration import robust_bank
from ku_betting_cs.laws import get_law

from cal_transfer import bank, calibrate, eighth_moment, load_calibration_config
from common import ROOT


def main() -> int:
    config = load_calibration_config()
    seeds = np.random.SeedSequence(config["seed"]).spawn(64)
    cap = config["penalty_cap"]
    rows = []
    for i, law_name in enumerate(config["CAL-1"]["laws"]):
        law = get_law(law_name)
        k8 = config["eighth_moment_envelope_factor"] * eighth_moment(law_name)
        for j, n_cal in enumerate(config["calibration_sizes"]):
            rng = np.random.default_rng(int(seeds[3 * i + j].generate_state(1)[0]))
            worst = 0.0
            for _ in range(config["CAL-1"]["replications"]):
                eps = law.sample(rng, n_cal)
                region, _, _, _ = calibrate(config, law, eps, k8)
                for scales in (bank(config, True), bank(config, False)):
                    _, pen = robust_bank(region.fbar, region.radius, scales, cap)
                    worst = max(worst, float(pen.max()))
                law.sample(rng, config["CAL-1"]["horizon"])
                for _ in config["CAL-3"]["signals"]:
                    law.sample(rng, config["CAL-3"]["horizon"])
            rows.append({"law": law_name, "n_cal": n_cal, "max_penalty": worst,
                         "cap_reached": worst >= cap})
            print(law_name, n_cal, f"max penalty {worst:.4f}", flush=True)
    payload = {"cap": cap, "rows": rows,
               "any_cap_reached": any(r["cap_reached"] for r in rows)}
    (ROOT / "results" / "cal_penalty_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
