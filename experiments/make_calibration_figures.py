"""Figure for the CAL study, generated only from its JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MARKERS = {"gamma2": "o", "lognormal04": "s", "ziln_p60": "^"}
LABELS = {"gamma2": "Gamma(2)", "lognormal04": "lognormal", "ziln_p60": "ZILN(0.60)"}
COLORS = {10_000: "C0", 100_000: "C1", 1_000_000: "C2"}


def width_law() -> None:
    rows = json.loads((ROOT / "results" / "cal_transfer_2.json").read_text())["rows"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for law, marker in MARKERS.items():
        for n_cal, color in COLORS.items():
            sub = sorted((r for r in rows if r["law"] == law and r["n_cal"] == n_cal),
                         key=lambda r: r["n"])
            left.plot([r["n"] for r in sub], [r["mean_ratio_robust_linear"] for r in sub],
                      marker=marker, color=color, linewidth=1)
        oracle = np.mean([r["mean_ratio_oracle"] for r in rows if r["law"] == law])
        left.plot([100, 10000], [oracle, oracle], linestyle=":", color="0.4",
                  linewidth=0.8)
    left.axhline(1.0, color="black", linewidth=0.8)
    left.set_xscale("log")
    left.set_xlabel("monitored length $n$")
    left.set_ylabel("width ratio, calibrated moment / first order")
    for n_cal, color in COLORS.items():
        left.plot([], [], color=color, label=f"$N=10^{{{len(str(n_cal)) - 1}}}$")
    for law, marker in MARKERS.items():
        left.plot([], [], marker, color="0.3", linestyle="none", label=LABELS[law])
    left.plot([], [], ":", color="0.4", label="oracle ratio")
    left.legend(frameon=False, fontsize=7, ncol=4, loc="lower center",
                bbox_to_anchor=(0.5, 1.0))

    pred = np.array([r["mean_predicted_ratio"] for r in rows])
    emp = np.array([r["mean_ratio_robust_linear"] for r in rows])
    for law, marker in MARKERS.items():
        mask = np.array([r["law"] == law for r in rows])
        right.plot(pred[mask], emp[mask], marker, linestyle="none", color="0.25",
                   markerfacecolor="none", label=LABELS[law])
    lo, hi = min(pred.min(), emp.min()) - 0.01, max(pred.max(), emp.max()) + 0.01
    right.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    right.set_xlabel("matched-boundary prediction")
    right.set_ylabel("observed mean width ratio")
    right.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "cal_width_law.pdf", bbox_inches="tight", dpi=220)
    fig.savefig(out / "cal_width_law.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    width_law()
