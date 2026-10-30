"""BET-1: Monte Carlo audit of time-uniform type-I error.

The mathematical guarantee is Ville's inequality.  This gate checks the
implementation against it and uses a one-sided 95% Clopper--Pearson upper
bound fixed in configs/gates.json before this run.
"""

from __future__ import annotations

import numpy as np

from ku_betting_cs.core import log_mixture_path
from ku_betting_cs.laws import get_law
from ku_betting_cs.stats import clopper_pearson_upper

from common import jsonable, load_config, scale_bank, write_json


def main() -> int:
    config = load_config()
    gate = config["BET-1"]
    rng = np.random.default_rng(config["seed"])
    scales = scale_bank(config, two_sided=True)
    threshold = np.log(1.0 / config["alpha"])
    rows = []
    passed = True
    for law_name in gate["laws"]:
        law = get_law(law_name)
        hits = 0
        for start in range(0, gate["replications"], gate["batch"]):
            batch = min(gate["batch"], gate["replications"] - start)
            x = law.sample(rng, (batch, gate["horizon"]))
            loge = log_mixture_path(
                x, 0.0, law.mu3, law.mu4, "polynomial", scales
            )
            hits += int(np.sum(np.max(loge, axis=1) >= threshold))
        rate = hits / gate["replications"]
        upper = clopper_pearson_upper(hits, gate["replications"])
        row_pass = upper <= gate["cp_upper_max"]
        passed &= row_pass
        rows.append(
            {
                "law": law_name,
                "g2": law.g2,
                "hits": hits,
                "replications": gate["replications"],
                "rate": rate,
                "cp95_upper": upper,
                "pass": row_pass,
            }
        )
        print(law_name, hits, f"rate={rate:.4f}", f"upper={upper:.4f}", row_pass)
    payload = {"gate": "BET-1", "pass": passed, "rows": rows, "config": gate}
    write_json("bet1_validity.json", jsonable(payload))
    print("BET-1", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
