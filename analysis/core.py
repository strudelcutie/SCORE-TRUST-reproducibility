from __future__ import annotations
import math
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('DILITRUST_DATA', REPO)).resolve()
DILI_VALID_DIR = DATA / 'analysis' / 'inputs' / 'dili_validation'
DILI_TEST = DATA / 'results' / 'predictions' / 'prediction_scores_388.csv'
RELEASE_DILI = DATA / 'results' / 'figure_data' / 'dili' / 'canonical' / 'predictions' / 'histgb_predictions.csv'
CROSS = DATA / 'results' / 'predictions' / 'cross_domain'
NASA_PRED = CROSS / 'nasa'
OUT = Path(os.environ.get('DILITRUST_OUT', REPO / 'analysis' / 'output_tables')).resolve()
TABLES = OUT
INTER = OUT
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [20260521, 20260522, 20260523, 20260524, 20260525]
ALPHA = 0.1

@dataclass
class Setting:
    key: str
    domain: str
    label: str
    test_path: Path
    valid_path: Path
    ad_path: Path | None
    th_spec95: float
    th_sens95: float
    u80: float
    qhat: float
    calib: str
CONFIG: dict[str, Setting] = {}
for key, label, th_spec, th_sens, u80, qhat in [('dili_scaffold', 'DILI scaffold', 0.828, 0.4273, 1.082826, 0.7328), ('dili_external', 'DILI external', 0.828, 0.4273, 1.082826, 0.7328), ('dili_temporal', 'DILI temporal', 0.8313, 0.4015, 1.088796, 0.7281), ('dili_cluster', 'DILI cluster-out', 0.7953, 0.262, 1.077283, 0.694)]:
    valid_split = 'scaffold' if key == 'dili_external' else {'dili_scaffold': 'scaffold', 'dili_temporal': 'temporal', 'dili_cluster': 'cluster_out'}[key]
    CONFIG[key] = Setting(key, 'DILI', label, RELEASE_DILI, DILI_VALID_DIR / f'{valid_split}_HistGB_locked_calibrated_predictions.csv', None, th_spec, th_sens, u80, qhat, 'precomputed')
for key, label, assay, split, th_spec, th_sens, u80, qhat in [('tox_nrar_scaffold', 'Tox21 NR-AR scaffold', 'nrar', 'scaffold', 0.0654, 0.0063, 0.174631, 0.055), ('tox_nrar_cluster', 'Tox21 NR-AR cluster-out', 'nrar', 'cluster_out', 0.0752, 0.013, 0.192243, 0.0633), ('tox_nrer_scaffold', 'Tox21 NR-ER scaffold', 'nrer', 'scaffold', 0.226, 0.0348, 0.545201, 0.546), ('tox_nrer_cluster', 'Tox21 NR-ER cluster-out', 'nrer', 'cluster_out', 0.1783, 0.0572, 0.485284, 0.7208)]:
    base = CROSS / f'tox21_{assay}' / 'predictions'
    CONFIG[key] = Setting(key, 'Tox21', label, base / f'{split}_HistGB_locked_calibrated_test.csv', base / f'{split}_HistGB_locked_calibrated_valid.csv', base / f'ad_features_{split}.csv', th_spec, th_sens, u80, qhat, 'precomputed')
for key, label, project in [('nasa_kc1', 'NASA KC1', 'kc1_internal'), ('nasa_pc1', 'NASA PC1', 'kc1_pc1'), ('nasa_jm1', 'NASA JM1', 'kc1_jm1'), ('nasa_kc2', 'NASA KC2', 'kc1_kc2')]:
    base = NASA_PRED / 'predictions'
    CONFIG[key] = Setting(key, 'NASA', label, base / f'{project}_HistGB_locked_calibrated_test.csv', base / f'{project}_HistGB_locked_calibrated_valid.csv', base / f'ad_features_{project}.csv', 0.4354, 0.0512, 0.760856, 0.6902, 'precomputed')

def _logit(p: np.ndarray, eps: float=1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def fit_temperature(s: np.ndarray, y: np.ndarray) -> float:
    z = _logit(s)
    grid = np.exp(np.linspace(np.log(0.02), np.log(50.0), 400))

    def nll(T: float) -> float:
        p = _sigmoid(z / T)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    losses = [nll(t) for t in grid]
    return float(grid[int(np.argmin(losses))])

def apply_temperature(s: np.ndarray, T: float) -> np.ndarray:
    return _sigmoid(_logit(s) / T)

def fit_platt(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    z = _logit(s).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(z, y)
    return (float(lr.coef_[0][0]), float(lr.intercept_[0]))

def apply_platt(s: np.ndarray, A: float, B: float) -> np.ndarray:
    return _sigmoid(A * _logit(s) + B)

def calibrate(s: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    if method == 'temperature':
        return apply_temperature(s, fit_temperature(s, y))
    if method == 'platt':
        A, B = fit_platt(s, y)
        return apply_platt(s, A, B)
    raise ValueError(method)

def _load_dili(key: str) -> dict:
    valid_split = 'scaffold' if key == 'dili_external' else {'dili_scaffold': 'scaffold', 'dili_temporal': 'temporal', 'dili_cluster': 'cluster_out'}[key]
    f = DILI_VALID_DIR / f'{valid_split}_HistGB_locked_calibrated_predictions.csv'
    v = pd.read_csv(f).rename(columns={'raw_score': 'raw', 'calibrated_score': 'cal'})
    v = v[['sample_id', 'y_true', 'raw', 'cal']].copy()
    v['ood'] = False
    t = pd.read_csv(RELEASE_DILI)
    t = t[t['split'] == valid_split if key != 'dili_external' else t['split'] == 'external_dilist']
    t = t[t['model'] == 'HistGB'].rename(columns={'raw_score': 'raw', 'calibrated_probability': 'cal'}).copy()
    t = t[['sample_id', 'y_true', 'raw', 'cal']]
    if key == 'dili_external':
        ps = pd.read_csv(DILI_TEST)
        ps = ps[(ps['split_name'] == 'external_dilist') & (ps['model'] == 'HistGB')]
        m = dict(zip(ps['compound_id_anonymized'], ps['ood_flag']))
        t['ood'] = t['sample_id'].map(m).fillna(False).astype(bool)
    else:
        ps = pd.read_csv(DILI_TEST)
        name_map = {'dili_scaffold': 'scaffold', 'dili_temporal': 'temporal', 'dili_cluster': 'cluster_out'}
        ps = ps[(ps['split_name'] == name_map[key]) & (ps['model'] == 'HistGB')]
        m = dict(zip(ps['compound_id_anonymized'], ps['ood_flag']))
        t['ood'] = t['sample_id'].map(m).fillna(False).astype(bool)
    return {'valid': v, 'test': t}

def _load_precomputed(st: Setting) -> dict:
    v = pd.read_csv(st.valid_path).rename(columns={'raw_score': 'raw', 'calibrated_score': 'cal'})
    t = pd.read_csv(st.test_path).rename(columns={'raw_score': 'raw', 'calibrated_score': 'cal'})
    v = v[['sample_id', 'y_true', 'raw', 'cal']].copy()
    t = t[['sample_id', 'y_true', 'raw', 'cal']].copy()
    if st.ad_path is not None and st.ad_path.exists():
        ad = pd.read_csv(st.ad_path)[['sample_id', 'ood_extreme_flag']]
        t = t.merge(ad, on='sample_id', how='left')
        t['ood'] = t['ood_extreme_flag'].fillna(False).astype(bool)
        v = v.merge(ad, on='sample_id', how='left')
        v['ood'] = v['ood_extreme_flag'].fillna(False).astype(bool)
    else:
        t['ood'] = False
        v['ood'] = False
    return {'valid': v, 'test': t}

def load_setting(key: str) -> dict:
    st = CONFIG[key]
    if st.domain == 'DILI':
        d = _load_dili(key)
    else:
        d = _load_precomputed(st)
    d['setting'] = st
    return d

def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def uncertainty(p: np.ndarray) -> np.ndarray:
    return entropy(p) + 0.5 * (1 - np.abs(p - 0.5) * 2)

def threshold_at_spec(scores: np.ndarray, y: np.ndarray, spec: float=0.95) -> float:
    fpr, _, thr = roc_curve(y, scores)
    idx = np.where(1.0 - fpr >= spec)[0]
    if len(idx) == 0:
        return float('nan')
    return float(thr[idx[-1]])

def threshold_at_sens(scores: np.ndarray, y: np.ndarray, sens: float=0.95) -> float:
    _, tpr, thr = roc_curve(y, scores)
    idx = np.where(tpr >= sens)[0]
    if len(idx) == 0:
        return float('nan')
    return float(thr[idx[0]])

def conformal_qhat(scores: np.ndarray, y: np.ndarray, alpha: float=ALPHA) -> float:
    p = np.where(y == 1, scores, 1 - scores)
    r = np.sort(1 - p)
    n = len(r)
    k = math.ceil((n + 1) * (1 - alpha))
    k = min(max(k, 1), n)
    return float(r[k - 1])

def conformal_set(p: float, qhat: float) -> str:
    if qhat >= 0.5:
        if p > qhat:
            return 'positive'
        if p < 1 - qhat:
            return 'negative'
        return 'both'
    if p >= 1 - qhat:
        return 'positive'
    if p <= qhat:
        return 'negative'
    return 'empty'

def wilson(k: int, n: int, z: float=1.96) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))

def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def ece(y: np.ndarray, p: np.ndarray, bins: int=5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    n = len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() > 0:
            tot += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(tot)

def apply_protocol(test: pd.DataFrame, threshold_state: dict, flags: dict) -> pd.DataFrame:
    p = test[threshold_state['score_col']].values
    th_spec = threshold_state['th_spec95']
    th_sens = threshold_state['th_sens95']
    qhat = threshold_state['qhat']
    u80 = threshold_state['u80']
    ood = test['ood'].values.astype(bool)
    use_B = flags['B']
    use_C = flags['C']
    use_D = flags['D']
    use_E = flags['E']
    n = len(test)
    decision = np.array(['no_call'] * n, dtype=object)
    abst_c = np.zeros(n, dtype=bool)
    abst_u = np.zeros(n, dtype=bool)
    for i in range(n):
        if use_E and ood[i]:
            decision[i] = 'ood'
            continue
        if use_C:
            s = conformal_set(p[i], qhat)
            if s in ('both', 'empty'):
                abst_c[i] = True
                decision[i] = 'no_call'
                continue
        if use_D and uncertainty(np.array([p[i]]))[0] > u80:
            abst_u[i] = True
            decision[i] = 'no_call'
            continue
        if not use_B:
            decision[i] = 'call'
            continue
        if p[i] >= th_spec:
            decision[i] = 'high'
        elif p[i] < th_sens:
            if use_C:
                s = conformal_set(p[i], qhat)
                decision[i] = 'low' if s == 'negative' else 'no_call'
            else:
                decision[i] = 'low'
        else:
            decision[i] = 'no_call'
    out = test.copy()
    out['decision'] = decision
    out['abstain_conformal'] = abst_c
    out['abstain_u80'] = abst_u
    out['p_used'] = p
    return out

def summarise(dec: pd.DataFrame, with_groups: bool=True) -> dict:
    y = dec['y_true'].values.astype(int)
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    p = dec['p_used'].values
    res = {'n': n, 'positives': pos, 'prevalence': pos / n, 'auroc': auroc(y, p), 'brier': brier(y, p), 'ece': ece(y, p), 'n_ood': int((dec['decision'] == 'ood').sum()), 'n_abstain_conformal': int(dec['abstain_conformal'].sum()), 'n_abstain_u80': int(dec['abstain_u80'].sum()), 'n_no_call': int((dec['decision'] == 'no_call').sum()), 'no_call_pct': float((dec['decision'] == 'no_call').mean() * 100), 'n_called': int(dec['decision'].isin(['high', 'low', 'call']).sum())}
    if with_groups:
        hi = dec['decision'] == 'high'
        lo = dec['decision'] == 'low'
        res.update({'high_n': int(hi.sum()), 'high_tp': int(y[hi].sum()), 'high_fp': int((1 - y[hi]).sum()), 'high_precision': float(y[hi].mean()) if hi.sum() else float('nan'), 'high_recall': float(y[hi].sum() / pos) if pos else float('nan'), 'low_n': int(lo.sum()), 'low_pos': int(y[lo].sum()), 'low_neg': int((1 - y[lo]).sum()), 'npv': float((1 - y[lo]).mean()) if lo.sum() else float('nan'), 'fn': int(y[lo].sum()), 'fn_pct_pos': float(y[lo].sum() / pos * 100) if pos else float('nan'), 'low_coverage_pct': float(lo.mean() * 100)})
        k = int((1 - y[lo]).sum())
        m = int(lo.sum())
        lo_ci = wilson(k, m)
        res['npv_ci_low'], res['npv_ci_high'] = lo_ci
        base_neg = neg / n if n else float('nan')
        res['negative_share'] = float(k / m) if m else float('nan')
        res['negative_enrichment'] = float(k / m / base_neg) if m and base_neg else float('nan')
        called = dec['decision'].isin(['high', 'low', 'call'])
        res['retained_error_rate'] = float((1 - y[called]).mean()) if called.sum() else float('nan')
    return res
