"""Summarise the packet-level validation: per-seed means, paired exact Wilcoxon tests, and
comparison with the MATLAB results. usage: analyse.py <out dir> [json out]"""
import sys, glob, json, re, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
OUT = sys.argv[1]
METHODS = ['B1', 'B2', 'Proposed', 'Hybrid', 'HeurMC']
COLS = {'delay': 'delay', 'pktdelay': 'pktdelay', 'pdr': 'pdr', 'thr': 'thr', 'fair': 'fair',
        'loadvar': 'loadvar', 'handover': 'handover'}
res = {}
for L in sorted(os.listdir(OUT)):
    if not L.startswith('L'): continue
    per = {}
    for m in METHODS:
        S = {}
        for f in glob.glob(f'{OUT}/{L}/{m}_s*_e*.csv'):
            s, e = map(int, re.findall(r'_s(\d+)_e(\d+)', f)[0])
            d = pd.read_csv(f)
            S.setdefault(s, []).append(d.mean())
        if not S: continue
        per[m] = pd.DataFrame({s: pd.DataFrame(v).mean() for s, v in sorted(S.items())}).T
        per[m]['n_eps'] = [len(S[s]) for s in sorted(S)]
    res[L] = per

def p_exact(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0): return 1.0
    return float(wilcoxon(a, b, method='exact').pvalue)

summary = {}
for L, per in res.items():
    print(f'\n===== load scale {L[1:]} =====')
    summary[L] = {}
    for m, df in per.items():
        row = {k: (float(df[c].mean()), float(df[c].std(ddof=1)) if len(df) > 1 else 0.0)
               for k, c in COLS.items()}
        row['model_delay'] = (float(df['m_delay'].mean()), 0.0)
        row['model_thr'] = (float(df['m_thr'].mean()), 0.0)
        row['seeds'] = int(len(df)); row['episodes'] = int(df['n_eps'].sum())
        summary[L][m] = row
        print(f"{m:9s} n={len(df)} delay {row['delay'][0]:7.1f}±{row['delay'][1]:6.1f}  pktdelay {row['pktdelay'][0]:7.1f}  "
              f"pdr {row['pdr'][0]:.3f}  thr {row['thr'][0]:6.2f}±{row['thr'][1]:5.2f}  fair {row['fair'][0]:.3f}  "
              f"lv {row['loadvar'][0]:6.2f}  ho {row['handover'][0]:.4f} | model delay {row['model_delay'][0]:6.1f} thr {row['model_thr'][0]:5.1f}")
    if 'Hybrid' in per and len(per['Hybrid']) > 1:
        summary[L]['tests'] = {}
        for ref in ['Hybrid', 'Proposed']:
            for m in METHODS:
                if m == ref or m not in per: continue
                a, b = per[ref], per[m]
                common = sorted(set(a.index) & set(b.index))
                t = {k: p_exact(a.loc[common, c], b.loc[common, c]) for k, c in COLS.items()}
                summary[L]['tests'][f'{ref}_vs_{m}'] = t
                print(f"  {ref} vs {m}: " + "  ".join(f"{k} p={v:.4f}" for k, v in t.items()))
if len(sys.argv) > 2:
    json.dump(summary, open(sys.argv[2], 'w'), indent=1)
