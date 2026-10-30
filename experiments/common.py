from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "gates.json").read_text())


def scale_bank(config: dict, two_sided: bool = True) -> np.ndarray:
    alpha = float(config["alpha"])
    horizons = np.asarray(config["scale_horizons"], dtype=float)
    positive = np.sqrt(2.0 * np.log(2.0 / alpha) / horizons)
    if two_sided:
        return np.concatenate([-positive[::-1], positive])
    return positive


def write_json(name: str, payload: dict) -> None:
    path = ROOT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
