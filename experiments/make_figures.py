"""Generate manuscript figures only from frozen JSON gate outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", dpi=220)
    fig.savefig(FIGURES / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def validity() -> None:
    rows = load("bet1_validity.json")["rows"]
    x = np.arange(len(rows))
    rate = np.array([r["rate"] for r in rows])
    upper = np.array([r["cp95_upper"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    ax.errorbar(x, rate, yerr=[np.zeros_like(rate), upper - rate], fmt="o", capsize=4)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label=r"$\alpha=0.05$")
    ax.axhline(0.07, color="0.5", linestyle=":", linewidth=1, label="registered gate")
    ax.set_xticks(x, [r["law"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("ever-crossing probability")
    ax.set_ylim(0, 0.075)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "bet1_validity.pdf")


def widths() -> None:
    rows = load("bet2_width.json")["rows"]
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for law in sorted({r["law"] for r in rows}):
        rr = [r for r in rows if r["law"] == law]
        ax.plot([r["n"] for r in rr], [r["mean_ratio_poly_linear"] for r in rr], "o-", label=law)
        ax.plot([r["n"] for r in rr], [r["predicted_ratio"] for r in rr], linestyle=":", color=ax.lines[-1].get_color())
    ax.set_xscale("log")
    ax.set_xlabel("sample size")
    ax.set_ylabel("width ratio: moment / linear")
    ax.set_ylim(0.7, 1.04)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "bet2_width_ratio.pdf")


def growth() -> None:
    rows = load("bet4_growth.json")["rows"]
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for law in sorted({r["law"] for r in rows}):
        rr = [r for r in rows if r["law"] == law]
        ax.plot([r["delta"] for r in rr], [r["relative_error"] for r in rr], "o-", label=law)
    ax.axhline(0.1, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"location shift $\delta/\sigma$")
    ax.set_ylabel("relative quadratic-growth error")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "bet4_growth_error.pdf")


def ab() -> None:
    rows = load("bet5_ab.json")["rows"]
    labels = [f"{r['law']}\n{r['delta']:.2f}" for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for j, method in enumerate(["polynomial", "linear", "catoni"]):
        ax.bar(x + (j - 1) * width, [r["median_stop"][method] for r in rows], width, label=method)
    ax.set_xticks(x, labels)
    ax.set_ylabel("median stopping time")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save(fig, "bet5_stopping_time.pdf")


def main() -> None:
    validity()
    widths()
    growth()
    ab()
    print("generated 4 PDF and 4 PNG figures")


if __name__ == "__main__":
    main()
