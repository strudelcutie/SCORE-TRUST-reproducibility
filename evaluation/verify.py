"""Recompute the Table S17A counts for all 12 settings.

Parameters come from the locked values in ``analysis/core.py`` (CONFIG) and the
predictions come from the released files, so no parameter is read from the
workbook. Only the expected counts are read, from the all-components-on cell
(config ``11111``) of S17A; the check therefore does not verify the table
against itself.

Run from the repository root:

    python evaluation/verify.py

Writes ``evaluation/verification.csv`` with the observed and expected counts for
the high-score tier, the low-score group, and the positives left in that group.
"""

import argparse
import csv
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import core  # noqa: E402
from decision_rules import audit_gates  # noqa: E402

S17_SHEET = "S17_Abstention_Overlap"
ALL_ON = "11111"


def read_s17a(path):
    """Expected counts for the all-components-on cell, keyed by setting."""
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    if S17_SHEET not in wb.sheetnames:
        raise SystemExit(f"{path} has no sheet {S17_SHEET!r}")
    header = None
    expected = {}
    for row in wb[S17_SHEET].iter_rows(values_only=True):
        first = row[0] if row else None
        if first == "setting":
            header = list(row)
            continue
        if header is None:
            continue
        if isinstance(first, str) and first.startswith("S17"):
            break
        if first is None:
            continue
        rec = dict(zip(header, row))
        if str(rec.get("config")) != ALL_ON:
            continue
        expected[first] = (int(rec["high_n"]), int(rec["low_n"]), int(rec["fn"]))
    wb.close()
    return expected


def replay(key):
    """High-score tier, low-score group, and low-score positives under the full protocol."""
    data = core.load_setting(key)
    setting = data["setting"]
    test = data["test"]
    test = test[~test["ood"].astype(bool)]
    result = audit_gates(
        test["cal"].values,
        setting.qhat,
        setting.u80,
        setting.th_spec95,
        setting.th_sens95,
        test["y_true"].values,
    )
    return (result["n_high_retained"], result["n_low"], result["n_low_positive"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-s1", default=str(ROOT / "results" / "Data S1.xlsx"))
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "verification.csv"))
    args = ap.parse_args()

    expected = read_s17a(args.data_s1)
    rows = []
    for key, setting in core.CONFIG.items():
        high, low, fn = replay(key)
        exp = expected.get(key)
        if exp is None:
            rows.append([setting.label, high, "", low, "", fn, "", False])
            continue
        passed = (high, low, fn) == exp
        rows.append([setting.label, high, exp[0], low, exp[1], fn, exp[2], passed])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "setting",
                "high_n",
                "expected_high_n",
                "low_n",
                "expected_low_n",
                "low_positive",
                "expected_low_positive",
                "pass",
            ]
        )
        w.writerows(rows)

    ok = sum(1 for r in rows if r[-1])
    print(f"{ok}/{len(rows)} PASS")
    for r in rows:
        if not r[-1]:
            print("FAIL", r)
    print("wrote", out)


if __name__ == "__main__":
    main()
