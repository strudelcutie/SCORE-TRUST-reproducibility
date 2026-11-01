from __future__ import annotations
import numpy as np
import pandas as pd
import core
from core import CONFIG, TABLES, ALPHA
LADDER = {'L0 thresholds only': {'A': True, 'B': True, 'C': False, 'D': False, 'E': False}, 'L1 + conformal': {'A': True, 'B': True, 'C': True, 'D': False, 'E': False}, 'L2 + conformal + U80': {'A': True, 'B': True, 'C': True, 'D': True, 'E': False}, 'L3 full protocol': {'A': True, 'B': True, 'C': True, 'D': True, 'E': True}, 'thresholds + OOD only': {'A': True, 'B': True, 'C': False, 'D': False, 'E': True}, 'thresholds + conformal + OOD': {'A': True, 'B': True, 'C': True, 'D': False, 'E': True}}

def main() -> None:
    rows = []
    tox = []
    for key in CONFIG:
        d = core.load_setting(key)
        p = core.CONFIG[key]
        th_base = {'score_col': 'cal', 'th_spec95': p.th_spec95, 'th_sens95': p.th_sens95, 'qhat': p.qhat, 'u80': p.u80}
        for name, flags in LADDER.items():
            dec = core.apply_protocol(d['test'], th_base, flags)
            s = core.summarise(dec)
            decided = s['high_n'] + s['low_n']
            err = (s['high_fp'] + s['low_pos']) / decided if decided else np.nan
            rows.append({'setting': key, 'protocol': name, 'high_n': s['high_n'], 'high_precision': round(s['high_precision'], 4) if s['high_n'] else np.nan, 'high_recall': round(s['high_recall'], 4), 'low_n': s['low_n'], 'npv': round(s['npv'], 4) if s['low_n'] else np.nan, 'fn': s['fn'], 'fn_pct_pos': round(s['fn_pct_pos'], 2), 'no_call_pct': round(s['no_call_pct'], 2), 'decided_error_rate': round(err, 4) if decided else np.nan})
        if key.startswith(('tox', 'nasa')):
            no_abstain = {'A': True, 'B': True, 'C': False, 'D': False, 'E': True}
            dec_guard = core.apply_protocol(d['test'], th_base, no_abstain)
            dec_full = core.apply_protocol(d['test'], th_base, LADDER['L3 full protocol'])
            y = d['test']['y_true'].values.astype(int)
            guard_high = dec_guard['decision'] == 'high'
            full_high = dec_full['decision'] == 'high'
            removed = guard_high & ~full_high
            tox.append({'setting': key, 'high_before_abstention': int(guard_high.sum()), 'high_before_tp': int(y[guard_high].sum()), 'high_before_precision': round(y[guard_high].mean(), 4) if guard_high.sum() else np.nan, 'high_before_recall': round(y[guard_high].sum() / y.sum(), 4), 'base_rate': round(y.mean(), 4), 'high_after': int(full_high.sum()), 'removed_n': int(removed.sum()), 'removed_tp': int(y[removed].sum()), 'removed_fp': int((1 - y[removed]).sum()), 'removed_precision': round(y[removed].mean(), 4) if removed.sum() else np.nan, 'no_call_conformal': int(dec_full['abstain_conformal'].sum()), 'no_call_u80': int(dec_full['abstain_u80'].sum()), 'no_call_added_by_u80': int((dec_full['abstain_u80'] & ~dec_guard['abstain_u80']).sum())})
    pd.DataFrame(rows).to_csv(TABLES / 'S17E_protocol_ladder.csv', index=False)
    pd.DataFrame(tox).to_csv(TABLES / 'S17F_tier_loss_accounting.csv', index=False)
    ladder = pd.DataFrame(rows)
    print('=== protocol ladder (DILI) ===')
    print(ladder[ladder.setting.str.startswith('dili')][['setting', 'protocol', 'high_n', 'high_precision', 'high_recall', 'low_n', 'npv', 'fn', 'no_call_pct', 'decided_error_rate']].to_string())
    print('\n=== protocol ladder (Tox21 / NASA) ===')
    print(ladder[ladder.setting.str.startswith(('tox', 'nasa'))][['setting', 'protocol', 'high_n', 'high_precision', 'low_n', 'npv', 'fn', 'no_call_pct']].to_string())
    print('\n=== tier-loss accounting ===')
    print(pd.DataFrame(tox).to_string())
if __name__ == '__main__':
    main()
