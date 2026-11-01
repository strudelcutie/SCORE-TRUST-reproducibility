# Revision analyses

Code and output tables for the three methodological points in the resubmission.
Every path resolves inside this repository; nothing outside it is read.

## Run

```
pip install pandas numpy scikit-learn
python run_checks.py
python run_problem1.py
python run_problem2.py
python run_problem3.py
python run_problem3_supp.py
```

Outputs are written to `analysis/output_tables/`, overwriting the committed copies.

## Inputs

| File | Source |
|---|---|
| `analysis/inputs/dili_validation/*_locked_calibrated_predictions.csv` | calibration predictions on the DILI validation partitions |
| `results/figure_data/dili/canonical/predictions/histgb_predictions.csv` | DILI held-out test predictions |
| `results/predictions/prediction_scores_388.csv` | out-of-domain flags for the DILI cohorts |
| `results/predictions/cross_domain/tox21_*/predictions/` | Tox21 test, validation and applicability-domain files |
| `results/predictions/cross_domain/nasa/predictions/` | NASA test, validation and applicability-domain files |

The published locked parameters for all 12 settings are constants in `core.CONFIG`,
so no external threshold tables are needed.

## Environment variables

`DILITRUST_DATA` points at a different checkout of this repository.
`DILITRUST_OUT` changes the output directory.

## What each script tests

| Script | Question | Tables |
|---|---|---|
| `run_checks.py` | do the locked parameters reproduce from the validation partitions? | `parameter_reproduction.csv` |
| `run_problem1.py` | which low-score groups justify exclusion under prespecified criteria? | `Table5_revised_eligibility.csv`, `S8D_eligibility_sensitivity_grid.csv` |
| `run_problem2.py` | does separating the calibrator from the conformal quantile change coverage? | `S7A_conformal_validation_separation.csv`, `S7B_split_repeat_distribution.csv` |
| `run_problem3.py` | what does each protocol component contribute? | `S17A` to `S17D` |
| `run_problem3_supp.py` | what does each added component buy? | `S17E`, `S17F` |

## Pairing rule in `run_problem2.py`

Each estimator pairs its quantile with a matching test calibration:

- `shared`: quantile and test probabilities from the full validation fit.
- `cross`: quantile from out-of-fold validation scores, test probabilities from the
  full validation fit, the standard cross-conformal construction.
- `split60_median`: for each of 50 repeats the calibrator is fitted on 60% of
  validation, the quantile is read from the remaining 40%, and the test probabilities
  come from that same 60% fit. Every reported metric is the median of the paired
  per-repeat values. The `test_calibrator` column records the pairing.

## Reproduction check

`run_checks.py` rebuilds the quantile, both thresholds and the uncertainty cutoff from
each validation partition and compares them with `core.CONFIG`. The reconstructed
quantile matches the published value to within 0.003, the specificity and sensitivity
thresholds reproduce the published values exactly, and the uncertainty cutoff matches
to four decimals, in all 12 settings. `run_problem3.py` uses the locked parameters for
the calibrated cells and re-derives them only for the uncalibrated ones, so the
all-components-on cell of `S17A` reproduces the high-score and low-score counts of
Table 5 in the main text.
