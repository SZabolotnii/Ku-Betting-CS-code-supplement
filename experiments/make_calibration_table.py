"""Print the CAL-3 table rows and CAL-1 summary used in the manuscript."""

from __future__ import annotations

import json

from common import ROOT

NAMES = {"gamma2": "Gamma(2)", "lognormal04": "lognormal",
         "ziln_p60": "ZILN(0.60)", "ziln_p25": "ZILN(0.25)"}


def main() -> None:
    data = json.loads((ROOT / "results" / "cal_transfer_13.json").read_text())
    print("% CAL-3 rows: law & N & robust & linear & gain & oracle poly & oracle lin & oracle gain")
    for cell in data["cells"]:
        for alt in cell["alternatives"]:
            if abs(alt["delta"] - 0.12) > 1e-12:
                continue
            med = alt["median_stop"]
            print(f"{NAMES[cell['law']]} & $10^{{{len(str(cell['n_cal'])) - 1}}}$ & "
                  f"{med['robust']:.1f} & {med['linear']:.1f} & "
                  f"{100 * alt['median_gain_robust_vs_linear']:.1f}\\% & "
                  f"{med['oracle_poly']:.1f} & {med['oracle_lin']:.1f} & "
                  f"{100 * alt['median_gain_oracle_poly_vs_oracle_lin']:.1f}\\% \\\\")
    print("% CAL-1 summary")
    for cell in data["cells"]:
        nul = cell["null"]
        print(cell["law"], cell["n_cal"],
              "robust", nul["robust"]["crossings"], round(nul["robust"]["cp_upper"], 4),
              "linear", nul["linear"]["crossings"], round(nul["linear"]["cp_upper"], 4),
              "region_miss", cell["region_misses"], "lin_miss", cell["linear_bound_misses"],
              "gains", [round(a["median_gain_robust_vs_linear"], 3) for a in cell["alternatives"]])
    print("gates", data["gates"])


if __name__ == "__main__":
    main()
