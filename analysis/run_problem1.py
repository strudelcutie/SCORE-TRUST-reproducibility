from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import core
from core import CONFIG, TABLES, INTER
FULL = {'A': True, 'B': True, 'C': True, 'D': True, 'E': True}
N_MIN = 30
NPV_MIN = 0.9
ENRICH_MIN = 1.1
FN_PCT_MAX = 5.0
COVER_MIN = 5.0

def evidence_vector() -> pd.DataFrame:
    rows = []
    for key in CONFIG:
        d = core.load_setting(key)
        p = CONFIG[key]
        th = {'score_col': 'cal', 'th_spec95': p.th_spec95, 'th_sens95': p.th_sens95, 'qhat': p.qhat, 'u80': p.u80}
        dec = core.apply_protocol(d['test'], th, FULL)
        s = core.summarise(dec)
        pos = s['positives']
        n = s['n']
        neg = n - pos
        low_pos = s['low_pos']
        low_neg = s['low_neg']
        rows.append({'setting': key, 'domain': d['setting'].domain, 'label': d['setting'].label, 'n': n, 'positives': pos, 'prevalence': round(pos / n, 3), 'low_n': s['low_n'], 'low_positive': low_pos, 'low_negative': low_neg, 'npv': round(s['npv'], 4) if s['low_n'] else np.nan, 'npv_ci_low': round(s['npv_ci_low'], 4) if s['low_n'] else np.nan, 'npv_ci_high': round(s['npv_ci_high'], 4) if s['low_n'] else np.nan, 'negative_share': round(s['negative_share'], 4) if s['low_n'] else np.nan, 'base_negative_rate': round(neg / n, 4), 'negative_enrichment': round(s['negative_enrichment'], 4) if s['low_n'] else np.nan, 'fn': low_pos, 'fn_pct_pos': round(low_pos / pos * 100, 2) if pos else np.nan, 'low_coverage_pct': round(s['low_coverage_pct'], 2), 'breakeven_cost_ratio': round(low_neg / low_pos, 2) if low_pos else np.inf, 'odds_ratio_vs_cohort': round(low_neg / low_pos / (neg / pos), 3) if low_pos and pos else np.inf, 'high_n': s['high_n'], 'high_precision': round(s['high_precision'], 3) if s['high_n'] else np.nan, 'no_call_pct': round(s['no_call_pct'], 1)})
    return pd.DataFrame(rows)

def classify(row: pd.Series, n_min=N_MIN, npv_min=NPV_MIN, enrich_min=ENRICH_MIN, fn_max=FN_PCT_MAX, cover_min=COVER_MIN) -> str:
    if row['low_n'] < n_min:
        return 'insufficient'
    if row['npv'] < npv_min:
        return 'does not support'
    if not row['negative_enrichment'] >= enrich_min:
        return 'does not support'
    if row['fn_pct_pos'] > fn_max:
        return 'does not support'
    if row['low_coverage_pct'] < cover_min:
        return 'does not support'
    return 'supports'

def main() -> None:
    ev = evidence_vector()
    ev['eligibility'] = ev.apply(classify, axis=1)
    ev['basis'] = ev.apply(_basis, axis=1)
    ev.to_csv(TABLES / 'Table5_revised_eligibility.csv', index=False)
    grid = []
    for n_min, npv_min, enrich_min, fn_max, cover_min in itertools.product([20, 30, 40], [0.85, 0.9, 0.95], [1.05, 1.1, 1.15, 1.2], [5, 10, 20], [1, 5, 10]):
        cls = ev.apply(lambda r: classify(r, n_min, npv_min, enrich_min, fn_max, cover_min), axis=1)
        grid.append({'n_min': n_min, 'npv_min': npv_min, 'enrich_min': enrich_min, 'fn_max_pct': fn_max, 'coverage_min_pct': cover_min, 'n_supports': int((cls == 'supports').sum()), 'n_does_not_support': int((cls == 'does not support').sum()), 'n_insufficient': int((cls == 'insufficient').sum()), 'sets_supporting': ';'.join(ev.loc[cls == 'supports', 'setting'].tolist())})
    g = pd.DataFrame(grid)
    g.to_csv(TABLES / 'S8D_eligibility_sensitivity_grid.csv', index=False)
    print('=== evidence vector ===')
    print(ev.drop(columns=['basis']).to_string())
    print('\n=== base-case classification ===')
    print(ev[['setting', 'eligibility', 'basis']].to_string())
    print('\n=== sensitivity: how often does each setting support exclusion? ===')
    cnt = {}
    for _, r in g.iterrows():
        for s in str(r['sets_supporting']).split(';'):
            if s:
                cnt[s] = cnt.get(s, 0) + 1
    print(pd.Series(cnt).sort_values(ascending=False).to_string())
    print('\nfraction of grid runs with zero settings supporting:', round((g['n_supports'] == 0).mean(), 3))

def _basis(row: pd.Series) -> str:
    if row['low_n'] < N_MIN:
        return f"low group n={row['low_n']} < {N_MIN}"
    fails = []
    if row['npv'] < NPV_MIN:
        fails.append(f"NPV {row['npv']:.3f} < {NPV_MIN}")
    if not row['negative_enrichment'] >= ENRICH_MIN:
        fails.append(f"enrichment {row['negative_enrichment']:.3f} < {ENRICH_MIN}")
    if row['fn_pct_pos'] > FN_PCT_MAX:
        fails.append(f"FN {row['fn_pct_pos']:.1f}% > {FN_PCT_MAX}%")
    if row['low_coverage_pct'] < COVER_MIN:
        fails.append(f"coverage {row['low_coverage_pct']:.1f}% < {COVER_MIN}%")
    return 'meets all criteria' if not fails else '; '.join(fails)
if __name__ == '__main__':
    main()
