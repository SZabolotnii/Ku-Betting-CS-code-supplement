"""REAL: photon energy channels of the Fermi GBM detector as a reference record.

Each event of a GBM time-tagged-event file carries a pulse-height channel
(PHA, 0..127). A window of ``reference_seconds`` of events is the reference
record of unknown location; the next ``stream_length`` events are the
monitored stream. The first hour of the day is used only to fix the unit of
measurement and the eighth-moment envelope, and is never a window.

REAL-1 runs both calibrated e-processes on the unshifted stream at the null
value "same location as the reference" (the reference mean, whose standard
error is about 0.001 in these units) and counts crossings: this tests the
same-law assumption on real data as well as the code. REAL-3 adds a constant
shift to the same streams and compares one-sided stopping times.

The stability margin is declared from the reference alone: the moment region
is enlarged to cover the moments of each of ``margin_chunks`` consecutive
chunks of the reference window.

    PYTHONPATH=src python3 experiments/real_gbm.py [dev|formal]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import re
import sys
import urllib.request

import numpy as np

from ku_betting_cs.calibration import (
    MomentRegion,
    calibrate_moments_unknown_center,
    calibrate_variance_upper_unknown_center,
    dominating_matrix,
    robust_bank,
    robust_log_mixture_path,
)
from ku_betting_cs.core import crossing_times, log_mixture_path
from ku_betting_cs.stats import clopper_pearson_upper

from common import ROOT, jsonable

BASE = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily"
CACHE = ROOT / "data" / "gbm"


def load_real_config() -> dict:
    return json.loads((ROOT / "configs" / "real_gbm.json").read_text())


# ------------------------------------------------------------------- data

def fetch(day: str, detector: str, hours: list[int]) -> list[dict]:
    """Download (once) the TTE files of ``day`` (YYYY-MM-DD); return a manifest."""
    yyyy, mm, dd = day.split("-")
    folder = f"{BASE}/{yyyy}/{mm}/{dd}/current/"
    listing = urllib.request.urlopen(folder, timeout=60).read().decode()
    stem = f"glg_tte_{detector}_{yyyy[2:]}{mm}{dd}"
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = []
    for hour in hours:
        names = sorted(set(re.findall(rf"{stem}_{hour:02d}z_v\d\d\.fit\.gz", listing)))
        if not names:
            raise FileNotFoundError(f"{stem}_{hour:02d}z not listed")
        name = names[-1]  # highest version
        path = CACHE / name
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(folder + name, tmp)
            tmp.rename(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"hour": hour, "file": name, "url": folder + name, "sha256": digest})
    return manifest


def read_events(path) -> tuple[np.ndarray, np.ndarray]:
    """TIME and PHA columns of the EVENTS extension of a gzipped FITS file."""
    raw = gzip.open(path, "rb").read()
    pos = 0
    while pos < len(raw):
        header, done = {}, False
        while not done:
            block = raw[pos:pos + 2880]
            pos += 2880
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode("ascii")
                if card.startswith("END") and card[3:].strip() == "":
                    done = True
                    break
                if card[8:10] == "= ":
                    header[card[:8].strip()] = card[10:].split("/")[0].strip().strip("'").strip()
        size = 0
        if int(header.get("NAXIS", 0)):
            size = abs(int(header["BITPIX"])) // 8
            for i in range(1, int(header["NAXIS"]) + 1):
                size *= int(header[f"NAXIS{i}"])
            size += int(header.get("PCOUNT", 0))
        if header.get("EXTNAME") == "EVENTS":
            codes = {"D": ">f8", "I": ">i2", "J": ">i4", "E": ">f4", "B": "u1"}
            fields = []
            for i in range(1, int(header["TFIELDS"]) + 1):
                form = header[f"TFORM{i}"]
                rep = int(form[:-1] or 1)
                dtype = codes[form[-1]]
                fields.append((header[f"TTYPE{i}"], dtype) if rep == 1
                              else (header[f"TTYPE{i}"], dtype, (rep,)))
            rows = int(header["NAXIS2"]) * int(header["NAXIS1"])
            table = np.frombuffer(raw[pos:pos + rows], dtype=np.dtype(fields))

            def column(name: str) -> np.ndarray:
                # FITS column scaling: physical = TZERO + TSCAL * stored.
                i = next(j for j in range(1, int(header["TFIELDS"]) + 1)
                         if header[f"TTYPE{j}"] == name)
                return (float(header.get(f"TZERO{i}", 0.0))
                        + float(header.get(f"TSCAL{i}", 1.0)) * table[name].astype(float))

            return column("TIME"), column("PHA")
        pos += -(-size // 2880) * 2880
    raise ValueError(f"no EVENTS extension in {path}")


def load_hours(manifest: list[dict], hours: list[int]) -> tuple[np.ndarray, np.ndarray]:
    parts = [read_events(CACHE / m["file"]) for m in manifest if m["hour"] in hours]
    t = np.concatenate([p[0] for p in parts])
    x = np.concatenate([p[1] for p in parts])
    # Hourly files overlap by about two minutes; drop the repeated events.
    order = np.lexsort((x, t))
    t, x = t[order], x[order]
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = (np.diff(t) != 0.0) | (np.diff(x) != 0.0)
    return t[keep], x[keep]


def windows(t: np.ndarray, x: np.ndarray, cfg: dict) -> tuple[list, dict]:
    """Consecutive non-overlapping (reference, stream) pairs and an exclusion log."""
    w, n = cfg["reference_seconds"], cfg["stream_length"]
    candidates = []
    start = 0
    while True:
        stop = np.searchsorted(t, t[start] + w, side="left")
        if stop + n > t.size:
            break
        span = t[start:stop + n]
        candidates.append((start, stop, float(np.max(np.diff(span)))))
        start = stop + n
    counts = np.array([b - a for a, b, _ in candidates])
    floor = cfg["rate_floor"] * float(np.median(counts))
    kept, log = [], {"candidates": len(candidates), "low_rate": 0, "gap": 0}
    for a, b, gap in candidates:
        if b - a < floor:
            log["low_rate"] += 1
        elif gap > cfg["max_gap_seconds"]:
            log["gap"] += 1
        else:
            kept.append((a, b))
    log["kept"] = len(kept)
    # Bracket: events within bracket_seconds before and after the stream.
    pairs = []
    for a, b in kept:
        t_lo, t_hi = t[b] - cfg["bracket_seconds"], t[b + n - 1] + cfg["bracket_seconds"]
        before = x[np.searchsorted(t, t_lo):b]
        after = x[b + n:np.searchsorted(t, t_hi, side="right")]
        pairs.append((x[a:b], x[b:b + n], np.concatenate([before, after])))
    return pairs, log


# ---------------------------------------------------------------- analysis

def _central(z: np.ndarray) -> tuple[float, float, float]:
    d = z - z.mean()
    return float(np.mean(d**2)), float(np.mean(d**3)), float(np.mean(d**4))


def with_margin(region: MomentRegion, s_lin: float, ref: np.ndarray,
                chunks: int) -> tuple[MomentRegion, float]:
    """Enlarge the region to contain the moments of every reference chunk."""
    moments = np.array([_central(c) for c in np.array_split(ref, chunks)])
    lo = min(region.variance_lower, moments[:, 0].min())
    hi = max(region.variance_upper, moments[:, 0].max())
    center, radius = 0.5 * (lo + hi), 0.5 * (hi - lo)
    m3_lo = min(region.mu3_center - region.mu3_radius, moments[:, 1].min())
    m3_hi = max(region.mu3_center + region.mu3_radius, moments[:, 1].max())
    mu4 = max(region.mu4_upper, moments[:, 2].max())
    c3, r3 = 0.5 * (m3_lo + m3_hi), 0.5 * (m3_hi - m3_lo)
    fbar = dominating_matrix(center, radius, c3, r3, mu4)
    return (MomentRegion(center, radius, c3, r3, mu4, region.crude_mu4_upper, fbar),
            max(s_lin, hi))


def bank(alpha_e: float, horizons: list[int], two_sided: bool) -> np.ndarray:
    positive = np.sqrt(2.0 * np.log(2.0 / alpha_e) / np.asarray(horizons, dtype=float))
    return np.concatenate([-positive[::-1], positive]) if two_sided else positive


def _window(args: tuple) -> dict:
    cfg, ref, stream, bracket, loc, scale, k8 = args
    ref = (ref - loc) / scale
    stream = (stream - loc) / scale
    bracket = (bracket - loc) / scale
    alpha_e, cap = cfg["alpha_e"], cfg["penalty_cap"]
    fr = (cfg["location_fraction"], cfg["crude_fraction"])
    region = calibrate_moments_unknown_center(ref, k8, cfg["delta_split_moment"], *fr)
    s_lin = calibrate_variance_upper_unknown_center(ref, k8, cfg["delta_split_linear"], *fr)
    variants = {"calibrated": (region, s_lin),
                "margin": with_margin(region, s_lin, ref, cfg["margin_chunks"])}
    # theta0: the bracket mean (stream location); same_location: the reference mean.
    theta0, same = float(bracket.mean()), float(ref.mean())
    two = bank(alpha_e, cfg["scale_horizons"], True)
    one = bank(alpha_e, cfg["scale_horizons"], False)
    out = {"n_ref": int(ref.size), "n_bracket": int(bracket.size), "theta0": theta0,
           "stream_minus_ref": {"mean": float(stream.mean() - same),
                                "var_ratio": float(stream.var() / ref.var())},
           "stream_minus_bracket": float(stream.mean() - theta0)}
    for name, (reg, sl) in variants.items():
        lam2, pen2 = robust_bank(reg.fbar, reg.radius, two, cap)
        lam1, pen1 = robust_bank(reg.fbar, reg.radius, one, cap)
        row = {"radius": reg.radius, "mu3_radius": reg.mu3_radius, "g2_bar": reg.g2_bar,
               "s_lin": sl, "max_penalty": float(max(pen1.max(), pen2.max()))}
        for key, value in (("null_cross", theta0), ("same_location_cross", same)):
            moment = robust_log_mixture_path(stream, value, reg.center, lam2, pen2, reg.fbar)
            linear = log_mixture_path(stream / np.sqrt(sl), value / np.sqrt(sl), 0.0, 0.0,
                                      "linear", two)
            row[key] = {"moment": bool(moment.max() >= np.log(1 / alpha_e)),
                        "linear": bool(linear.max() >= np.log(1 / alpha_e))}
        stops = {}
        for delta in cfg["signals"]:
            x = stream + delta
            stops[str(delta)] = {
                "moment": int(crossing_times(robust_log_mixture_path(
                    x, theta0, reg.center, lam1, pen1, reg.fbar), alpha_e)),
                "linear": int(crossing_times(log_mixture_path(
                    x / np.sqrt(sl), theta0 / np.sqrt(sl), 0.0, 0.0, "linear", one), alpha_e)),
            }
        row["stops"] = stops
        out[name] = row
    return out


def run(day: str, cfg: dict, label: str) -> dict:
    hours = cfg["development_hours"] if label == "dev" else list(range(24))
    manifest = fetch(day, cfg["detector"], hours)
    t0, x0 = load_hours(manifest, [cfg["characterization_hour"]])
    loc, scale = float(x0.mean()), float(x0.std())
    z0 = (x0 - loc) / scale
    k8 = cfg["eighth_moment_envelope_factor"] * float(np.mean(z0**8))
    rest = [h for h in hours if h != cfg["characterization_hour"]]
    t, x = load_hours(manifest, rest)
    after = t > t0.max()  # windows never reuse characterization events
    pairs, exclusion = windows(t[after], x[after], cfg)
    jobs = [(cfg, r, s, b, loc, scale, k8) for r, s, b in pairs]
    with ProcessPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(_window, jobs, chunksize=4))
    summary = summarize(rows, cfg)
    payload = {"label": label, "day": day, "manifest": manifest,
               "characterization": {"loc": loc, "scale": scale, "k8": k8,
                                    "mu3": float(np.mean(z0**3)), "mu4": float(np.mean(z0**4)),
                                    "events": int(x0.size)},
               "exclusion": exclusion, "summary": summary, "windows": rows}
    return payload


def summarize(rows: list[dict], cfg: dict) -> dict:
    k = len(rows)
    out = {"windows": k}
    for name in ("calibrated", "margin"):
        null = {m: int(sum(r[name]["null_cross"][m] for r in rows)) for m in ("moment", "linear")}
        same = {m: int(sum(r[name]["same_location_cross"][m] for r in rows))
                for m in ("moment", "linear")}
        alt = {}
        for delta in cfg["signals"]:
            med = {m: float(np.median([r[name]["stops"][str(delta)][m] for r in rows]))
                   for m in ("moment", "linear")}
            alt[str(delta)] = {"median_stop": med,
                               "median_gain": 1.0 - med["moment"] / med["linear"]}
        out[name] = {
            "null_crossings": null,
            "null_cp_upper": {m: clopper_pearson_upper(v, k, cfg["cp_level"])
                              for m, v in null.items()},
            "same_location_crossings": same,
            "median_radius": float(np.median([r[name]["radius"] for r in rows])),
            "median_g2_bar": float(np.median([r[name]["g2_bar"] for r in rows])),
            "max_penalty": float(max(r[name]["max_penalty"] for r in rows)),
            "alternatives": alt,
        }
    out["stream_var_ratio_median_abs_dev"] = float(np.median(
        [abs(r["stream_minus_ref"]["var_ratio"] - 1) for r in rows]))
    out["sd_stream_minus_ref_mean"] = float(np.std([r["stream_minus_ref"]["mean"] for r in rows]))
    out["sd_stream_minus_bracket_mean"] = float(np.std([r["stream_minus_bracket"] for r in rows]))
    out["median_bracket_events"] = float(np.median([r["n_bracket"] for r in rows]))
    return out


def main() -> int:
    cfg = load_real_config()
    what = sys.argv[1] if len(sys.argv) > 1 else "formal"
    day = cfg["development_day"] if what == "dev" else cfg["confirmation_day"]
    payload = run(day, cfg, what)
    s = payload["summary"]
    out = ROOT / "results" / ("real_gbm_dev.json" if what == "dev" else "real_gbm.json")
    passed = None
    if what == "formal":
        m = s["margin"]
        real1 = all(m["null_cp_upper"][k] <= cfg["REAL-1"]["cp_upper_max"]
                    for k in cfg["REAL-1"]["methods_gated"])
        real3 = (m["alternatives"][str(cfg["REAL-3"]["signal"])]["median_gain"]
                 >= cfg["REAL-3"]["required_median_gain"])
        passed = {"REAL-1": bool(real1), "REAL-3": bool(real3)}
        payload["gates"] = passed
    out.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(jsonable({"exclusion": payload["exclusion"], "summary": s,
                               "gates": passed}), indent=1))
    return 0 if passed is None or all(passed.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
