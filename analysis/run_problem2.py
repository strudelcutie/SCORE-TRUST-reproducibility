from __future__ import annotations
import numpy as np
import pandas as pd
import core
from core import CONFIG, TABLES, ALPHA
RNG = np.random.default_rng(20261101)
N_REPEATS = 50
K_FOLDS = 5
METRICS = ['coverage', 'pos_coverage', 'neg_coverage', 'avg_set_size', 'both_rate', 'empty_rate', 'no_call_pct', 'low_n', 'npv', 'fn', 'high_n']

def _method(key: str) -> str:
    if key.startswith('dili'):
        return 'platt' if key == 'dili_temporal' else 'temperature'
    return 'platt'

def _coverage(p, y, qhat):
    sets = [core.conformal_set(float(x), qhat) for x in p]
    covered = [s == f"{('positive' if yy else 'negative')}" or s == 'both' for s, yy in zip(sets, y)]
    pos = y == 1
    sizes = np.array([1 if s in ('positive', 'negative') else 0 if s == 'empty' else 2 for s in sets])
    return {'coverage': float(np.mean(covered)), 'pos_coverage': float(np.mean(np.array(covered)[pos])) if pos.any() else np.nan, 'neg_coverage': float(np.mean(np.array(covered)[~pos])) if (~pos).any() else np.nan, 'avg_set_size': float(sizes.mean()), 'both_rate': float(np.mean([s == 'both' for s in sets])), 'empty_rate': float(np.mean([s == 'empty' for s in sets]))}

def _triage(p, y, ood, th_spec, th_sens, qhat, u80):
    dec = pd.DataFrame({'y_true': y, 'cal': p, 'ood': ood})
    out = core.apply_protocol(dec, {'score_col': 'cal', 'th_spec95': th_spec, 'th_sens95': th_sens, 'qhat': qhat, 'u80': u80}, {'A': True, 'B': True, 'C': True, 'D': True, 'E': True})
    s = core.summarise(out)
    return {'no_call_pct': s['no_call_pct'], 'low_n': s['low_n'], 'npv': s['npv'], 'fn': s['fn'], 'high_n': s['high_n']}

def main() -> None:
    rows, dist = ([], [])
    for key in CONFIG:
        d = core.load_setting(key)
        st = d['setting']
        v, t = (d['valid'], d['test'])
        sv = v['raw'].values
        yv = v['y_true'].values.astype(int)
        stv = t['raw'].values
        yt = t['y_true'].values.astype(int)
        ood = t['ood'].values.astype(bool)
        method = _method(key)
        if method == 'temperature':
            T = core.fit_temperature(sv, yv)
            pv_shared = core.apply_temperature(sv, T)
            pt = core.apply_temperature(stv, T)
        else:
            A, B = core.fit_platt(sv, yv)
            pv_shared = core.apply_platt(sv, A, B)
            pt = core.apply_platt(stv, A, B)
        q_shared = core.conformal_qhat(pv_shared, yv, ALPHA)
        idx = RNG.permutation(len(sv))
        folds = np.array_split(idx, K_FOLDS)
        pv_oof = np.empty_like(sv)
        for k in range(K_FOLDS):
            hold = folds[k]
            fit = np.setdiff1d(idx, hold)
            if method == 'temperature':
                Tk = core.fit_temperature(sv[fit], yv[fit])
                pv_oof[hold] = core.apply_temperature(sv[hold], Tk)
            else:
                ak, bk = core.fit_platt(sv[fit], yv[fit])
                pv_oof[hold] = core.apply_platt(sv[hold], ak, bk)
        q_cross = core.conformal_qhat(pv_oof, yv, ALPHA)
        rep = []
        for _ in range(N_REPEATS):
            perm = RNG.permutation(len(sv))
            cut = int(0.6 * len(sv))
            fit, qs_part = (perm[:cut], perm[cut:])
            if method == 'temperature':
                T = core.fit_temperature(sv[fit], yv[fit])
                p_qs = core.apply_temperature(sv[qs_part], T)
                pt_r = core.apply_temperature(stv, T)
            else:
                a, b = core.fit_platt(sv[fit], yv[fit])
                p_qs = core.apply_platt(sv[qs_part], a, b)
                pt_r = core.apply_platt(stv, a, b)
            q = core.conformal_qhat(p_qs, yv[qs_part], ALPHA)
            rep.append({**_coverage(pt_r, yt, q), **_triage(pt_r, yt, ood, st.th_spec95, st.th_sens95, q, st.u80), 'qhat': q})
        rep_df = pd.DataFrame(rep)
        med = rep_df.median(numeric_only=True)
        base = {'setting': key, 'label': st.label, 'n_valid': len(sv), 'n_test': len(yt), 'method': method}
        cov_s = _coverage(pt, yt, q_shared)
        tri_s = _triage(pt, yt, ood, st.th_spec95, st.th_sens95, q_shared, st.u80)
        rows.append({**base, 'estimator': 'shared', 'qhat': round(q_shared, 4), 'qhat_source': 'full validation', 'test_calibrator': 'full validation (paired)', **{k: round(cov_s[k], 4) for k in METRICS[:6]}, **{k: round(v_, 4) if isinstance(v_, float) else v_ for k, v_ in tri_s.items()}})
        cov_c = _coverage(pt, yt, q_cross)
        tri_c = _triage(pt, yt, ood, st.th_spec95, st.th_sens95, q_cross, st.u80)
        rows.append({**base, 'estimator': 'cross', 'qhat': round(q_cross, 4), 'qhat_source': 'out-of-fold inside validation', 'test_calibrator': 'full validation (cross-conformal)', **{k: round(cov_c[k], 4) for k in METRICS[:6]}, **{k: round(v_, 4) if isinstance(v_, float) else v_ for k, v_ in tri_c.items()}})
        rows.append({**base, 'estimator': 'split60_median', 'qhat': round(float(med['qhat']), 4), 'qhat_source': '40% holdout of validation', 'test_calibrator': '60% fit, per repeat (paired)', **{k: round(float(med[k]), 4) for k in METRICS}})
        dist.append({**base, 'qhat_published': st.qhat, 'qhat_split_median': round(float(med['qhat']), 4), 'qhat_split_p05': round(float(rep_df['qhat'].quantile(0.05)), 4), 'qhat_split_p95': round(float(rep_df['qhat'].quantile(0.95)), 4), 'coverage_split_median': round(float(med['coverage']), 4), 'coverage_split_p05': round(float(rep_df['coverage'].quantile(0.05)), 4), 'coverage_split_p95': round(float(rep_df['coverage'].quantile(0.95)), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / 'S7A_conformal_validation_separation.csv', index=False)
    pd.DataFrame(dist).to_csv(TABLES / 'S7B_split_repeat_distribution.csv', index=False)
    print(out[['setting', 'estimator', 'qhat', 'coverage', 'pos_coverage', 'neg_coverage', 'no_call_pct', 'low_n', 'npv', 'fn']].to_string())
    print()
    print(pd.DataFrame(dist)[['setting', 'qhat_published', 'qhat_split_median', 'coverage_split_median', 'coverage_split_p05', 'coverage_split_p95']].to_string())
if __name__ == '__main__':
    main()
