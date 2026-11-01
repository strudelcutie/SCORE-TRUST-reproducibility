from __future__ import annotations
import numpy as np
import pandas as pd
import core
from core import CONFIG, ALPHA, INTER

def main() -> None:
    rows = []
    for key in CONFIG:
        d = core.load_setting(key)
        st = d['setting']
        v = d['valid']
        vc = v['cal'].values
        vy = v['y_true'].values.astype(int)
        rows.append({'setting': key, 'n_valid': len(v), 'n_test': len(d['test']), 'qhat_published': st.qhat, 'qhat_recomputed': round(core.conformal_qhat(vc, vy, ALPHA), 4), 'th_spec_published': st.th_spec95, 'th_spec_recomputed': round(core.threshold_at_spec(vc, vy, 0.95), 4), 'th_sens_published': st.th_sens95, 'th_sens_recomputed': round(core.threshold_at_sens(vc, vy, 0.95), 4), 'u80_published': st.u80, 'u80_recomputed': round(float(np.percentile(core.uncertainty(vc), 80)), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(INTER / 'parameter_reproduction.csv', index=False)
    print(out.to_string())
    print()
    print('wrote', INTER / 'parameter_reproduction.csv')
if __name__ == '__main__':
    main()
