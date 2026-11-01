from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import core
from core import CONFIG, TABLES, ALPHA
FACTORS = ['A', 'B', 'C', 'D', 'E']

def run_cell(d: dict, flags: dict) -> dict:
    v, t = (d['valid'], d['test'])
    score_col = 'cal' if flags['A'] else 'raw'
    sv = v[score_col].values
    yv = v['y_true'].values.astype(int)
    st = t[score_col].values
    yt = t['y_true'].values.astype(int)
    setting = d['setting']
    if flags['A']:
        th_spec, th_sens = (setting.th_spec95, setting.th_sens95)
        qhat = setting.qhat if flags['C'] else 0.5
        u80 = setting.u80 if flags['D'] else np.inf
    else:
        th_spec = core.threshold_at_spec(sv, yv, 0.95)
        th_sens = core.threshold_at_sens(sv, yv, 0.95)
        qhat = core.conformal_qhat(sv, yv, ALPHA) if flags['C'] else 0.5
        u80 = float(np.percentile(core.uncertainty(sv), 80)) if flags['D'] else np.inf
    th = {'score_col': score_col, 'th_spec95': th_spec, 'th_sens95': th_sens, 'qhat': qhat, 'u80': u80}
    dec = core.apply_protocol(t, th, flags)
    s = core.summarise(dec, with_groups=flags['B'])
    out = {'auroc': round(s['auroc'], 4), 'brier': round(s['brier'], 4), 'ece': round(s['ece'], 4), 'no_call_pct': round(s['no_call_pct'], 2), 'n_abstain_conformal': s['n_abstain_conformal'], 'n_abstain_u80': s['n_abstain_u80'], 'n_ood': s['n_ood']}
    if flags['B']:
        decided = s['high_n'] + s['low_n']
        errors = s['high_fp'] + s['low_pos']
        out.update({'high_n': s['high_n'], 'high_precision': round(s['high_precision'], 4) if s['high_n'] else np.nan, 'high_recall': round(s['high_recall'], 4), 'high_tp': s['high_tp'], 'high_fp': s['high_fp'], 'low_n': s['low_n'], 'low_positive': s['low_pos'], 'npv': round(s['npv'], 4) if s['low_n'] else np.nan, 'fn': s['fn'], 'fn_pct_pos': round(s['fn_pct_pos'], 2), 'low_coverage_pct': round(s['low_coverage_pct'], 2), 'decided_n': decided, 'decided_error_rate': round(errors / decided, 4) if decided else np.nan})
    return out

def main() -> None:
    all_rows = []
    for key in CONFIG:
        d = core.load_setting(key)
        for combo in itertools.product([False, True], repeat=5):
            flags = dict(zip(FACTORS, combo))
            r = run_cell(d, flags)
            all_rows.append({'setting': key, 'domain': d['setting'].domain, 'label': d['setting'].label, **{f: int(flags[f]) for f in FACTORS}, 'config': ''.join(('1' if flags[f] else '0' for f in FACTORS)), **r})
    out = pd.DataFrame(all_rows)
    out.to_csv(TABLES / 'S17A_factorial_ablation_32.csv', index=False)
    dec_block = out[out['B'] == 1].copy()
    effects = []
    for key, g in dec_block.groupby('setting'):
        for metric in ['high_n', 'high_precision', 'high_recall', 'low_n', 'npv', 'fn', 'no_call_pct', 'decided_error_rate', 'low_coverage_pct']:
            for f in ['A', 'C', 'D', 'E']:
                on = g[g[f] == 1][metric].astype(float)
                off = g[g[f] == 0][metric].astype(float)
                effects.append({'setting': key, 'metric': metric, 'factor': f, 'mean_on': round(on.mean(), 4) if len(on) else np.nan, 'mean_off': round(off.mean(), 4) if len(off) else np.nan, 'effect_on_minus_off': round(on.mean() - off.mean(), 4) if len(on) and len(off) else np.nan})
    eff = pd.DataFrame(effects)
    eff.to_csv(TABLES / 'S17B_component_main_effects.csv', index=False)
    q_eff = []
    for key, g in out.groupby('setting'):
        for metric in ['auroc', 'brier', 'ece']:
            for f in FACTORS:
                on = g[g[f] == 1][metric].astype(float)
                off = g[g[f] == 0][metric].astype(float)
                q_eff.append({'setting': key, 'metric': metric, 'factor': f, 'mean_on': round(on.mean(), 4), 'mean_off': round(off.mean(), 4), 'effect_on_minus_off': round(on.mean() - off.mean(), 4)})
    pd.DataFrame(q_eff).to_csv(TABLES / 'S17C_quality_main_effects.csv', index=False)
    nest = []
    for key in CONFIG:
        d = core.load_setting(key)
        v = d['valid']
        p = v['cal'].values
        yv = v['y_true'].values.astype(int)
        q = core.conformal_qhat(p, yv, ALPHA)
        u80 = float(np.percentile(core.uncertainty(p), 80))
        lo, hi = (0.0, 0.5)
        for _ in range(80):
            mid = (lo + hi) / 2
            if core.uncertainty(np.array([mid]))[0] < u80:
                lo = mid
            else:
                hi = mid
        u_lo = lo
        c_lo = 1 - q if q >= 0.5 else q
        c_hi = q if q >= 0.5 else 1 - q
        nest.append({'setting': key, 'qhat': round(q, 4), 'u80': round(u80, 4), 'u_band_lower': round(u_lo, 4), 'u_band_upper': round(1 - u_lo, 4), 'conformal_lower': round(c_lo, 4), 'conformal_upper': round(c_hi, 4), 'u_band_inside_conformal': bool(u_lo >= c_lo and 1 - u_lo <= c_hi)})
    pd.DataFrame(nest).to_csv(TABLES / 'S17D_u80_nesting.csv', index=False)
    print('=== 32-cell ablation written ===')
    print(out[out['setting'] == 'dili_external'][['config', 'A', 'B', 'C', 'D', 'E', 'auroc', 'no_call_pct', 'high_n', 'high_precision', 'low_n', 'npv', 'fn']].to_string())
    print('\n=== U80 band vs conformal band ===')
    print(pd.DataFrame(nest).to_string())
if __name__ == '__main__':
    main()
