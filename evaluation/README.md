`decision_rules.py` audits two abstention rules built on one calibrated probability.

Inputs: `p` (array), `qhat`, `U80`, `th_spec95`, `th_sens95`, and optionally the labels `y`.
Outputs: `p_star`, conformal and uncertainty edges, gate label, `tier_untouched_sufficient`, and, when `p` is given, the counts `n_high`, `n_high_removed`, `n_high_retained`, `n_low`, `n_low_positive` and `n_abstained`.
Assumes r = 1 - p and a common scale for qhat and the thresholds.
U80 < 0 is rejected; U80 = 0 is evaluated per sample as U(p) > U80.

Example:

```python
audit_gates(None, 0.7328, 1.082826, 0.828, 0.4273)["gate"]       # "conformal binds"
audit_gates(None, 0.055, 0.174631, 0.0654, 0.0063)["gate"]       # "uncertainty binds"
```

`verify.py` recomputes the Table S17A counts for all 12 settings. The locked
parameters come from `analysis/core.py` (CONFIG) and the predictions from the
released files; only the expected counts are read from the workbook, from the
all-components-on cell (config `11111`) of S17A. The audit therefore does not
check the table against itself.

```
python evaluation/verify.py
```

Writes `evaluation/verification.csv` with observed and expected counts for the
high-score tier, the low-score group, and the positives left in that group, and
prints `12/12 PASS`. `--data-s1` and `--out` override the default paths.

`locked_parameters.csv` lists the same locked values as `analysis/core.py` (CONFIG),
for readers who want the table without running Python.
