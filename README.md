# Ku-Betting-CS Code Supplement

Code and stored results for the manuscript

**Moment-Orthogonal Quadratic-Envelope E-Processes for Anytime-Valid Location Inference**
*Serhii Zabolotnii* (manuscript in preparation; a preprint link will be added here).

## The result in one line

For a location parameter with skewed errors, betting on the degree-two
stochastic polynomial score `(X - m, (X - m)^2 - sigma^2)` through the positive
quadratic envelope `(1 + h + h^2/2) / (1 + v/2)` gives an exact e-process on an
unbounded support, and its confidence sequence is narrower than the
first-order one by the factor `sqrt(g2)`, where
`g2 = 1 - gamma3^2 / (2 + gamma4)` is the variance-reduction coefficient of the
degree-two polynomial maximization method. When the moments are not known but
calibrated on an independent reference record, a centering-robust factor keeps
the guarantee exact and the remaining gain follows a closed-form width law.

## Start here

```sh
pip install -r requirements.txt
PYTHONPATH=src python -m pytest          # unit tests, a few seconds
```

Every number in the manuscript is read from a JSON file in `results/`. The
scripts below regenerate those files; each experiment's design (laws, sample
sizes, seeds and acceptance criteria) is fixed in a file under `configs/` that
was written before the experiment was run.

## Map from the manuscript to the code

| Manuscript | Script | Design | Stored result |
|---|---|---|---|
| Simulation 1: validity under the null | `experiments/bet1_validity.py` | `configs/gates.json` | `results/bet1_validity.json` |
| Simulation 2: width ratios | `experiments/bet2_width.py` | `configs/gates.json` | `results/bet2_width.json` |
| Simulation 3: projection identity | `experiments/bet3_kappa.py` | `configs/gates.json` | `results/bet3_kappa.json` |
| Simulation 4: quadratic growth approximation | `experiments/bet4_growth.py` | `configs/gates.json` | `results/bet4_growth.json` |
| Simulation 5: stopping times with known moments | `experiments/bet5_ab.py` | `configs/gates.json` | `results/bet5_ab.json` |
| Simulation 6: misspecified moments | `experiments/rev_misspec.py` | `configs/revision_gates.json` | `results/rev_misspec.json` |
| Simulation 7: betting strategies | `experiments/rev_strategy.py` | `configs/revision_gates.json` | `results/rev_strategy.json` |
| Simulation 8: calibrated validity and stopping times | `experiments/cal_transfer.py cal13` | `configs/calibration_gates.json` | `results/cal_transfer_13.json` |
| Simulation 9: calibrated width law | `experiments/cal_transfer.py cal2` | `configs/calibration_gates.json` | `results/cal_transfer_2.json` |
| Simulation 10: reference of unknown location | `experiments/cal_transfer.py cal4` | `configs/calibration_unknown_center.json` | `results/cal_unknown_center_4.json` |
| Gamma-ray detector example | `experiments/real_gbm.py formal` | `configs/real_gbm.json` | `results/real_gbm.json` |
| Same example, development day | `experiments/real_gbm.py dev` | `configs/real_gbm.json` | `results/real_gbm_dev.json` |
| Closed-form calibrated width (symbolic and brute-force check) | `experiments/verify_calibration_width.py` | — | `results/calibration_width_check.json` |
| Admissible quadratic family | `experiments/verify_quadratic_family.py` | — | `results/quadratic_family.json` |
| Penalty audit of the calibrated bank | `experiments/cal_penalty_audit.py` | — | `results/cal_penalty_audit.json` |

The file names keep the labels used while the study was run: `bet` for the main
study (Simulations 1–5), `rev` for the supplementary diagnostics (Simulations
6–7) and `cal` for the calibration study (Simulations 8–10).

## Rerunning

```sh
PYTHONPATH=src python run_all.py          # tests, theory checks, Simulations 1-5
PYTHONPATH=src python run_revision.py     # Simulations 6-7 and the quadratic family
PYTHONPATH=src python run_calibration.py  # Simulations 8-10, width-law check, figure
PYTHONPATH=src python experiments/real_gbm.py formal   # gamma-ray example
python experiments/make_figures.py        # remaining figures
```

The calibration study takes tens of minutes on a ten-core machine. The
gamma-ray example downloads about 450 MB per day of public Fermi Gamma-ray
Burst Monitor data (detector n0, time-tagged events) from the HEASARC archive
into `data/gbm/`, which is not tracked; the address and SHA-256 digest of every
file used are recorded in `results/real_gbm.json`.

## Layout

- `src/ku_betting_cs/` — the e-factor, mixture banks and confidence-sequence
  inversion (`core.py`), error laws (`laws.py`), Clopper–Pearson bounds
  (`stats.py`), and the calibration region, centering-robust factor and
  horizon-matched bank (`calibration.py`).
- `experiments/` — one script per experiment, plus figure and table generators.
- `configs/`, `results/`, `figures/` — designs, stored outputs, figures.
- `tests/` — unit tests, including exact e-variable checks on finite laws.
- `verify_*.py` — numerical sanity checks of the width and stitching results.
- `BettingCS.lean` — Lean 4 file for the algebraic projection and positivity
  lemmas; it compiles with Lean/Mathlib 4.26.0 without `sorry`
  (`lake env lean BettingCS.lean` inside a Mathlib project).

## Environment

The stored results were produced with Python 3.14.5, NumPy 2.4.4, SciPy 1.17.1
and Matplotlib 3.10.9 (`requirements.txt`).

## Licence

MIT, see `LICENSE`.
