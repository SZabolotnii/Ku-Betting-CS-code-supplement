"""Helpers for supplementary experiments, kept separate from frozen BET gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_revision_config() -> dict:
    return json.loads((ROOT / "configs" / "revision_gates.json").read_text())


def revision_scale_bank(config: dict, dense: int | None = None) -> np.ndarray:
    alpha = float(config["alpha"])
    if dense is None:
        horizons = np.asarray(config["scale_horizons"], dtype=float)
    else:
        horizons = np.geomspace(
            min(config["scale_horizons"]), max(config["scale_horizons"]), dense
        )
    positive = np.sqrt(2.0 * np.log(2.0 / alpha) / horizons)
    return np.concatenate([-positive[::-1], positive])


def write_revision_json(name: str, payload: dict) -> None:
    path = ROOT / "results" / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
