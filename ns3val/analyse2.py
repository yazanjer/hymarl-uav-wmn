"""Summarise the extension batch (cells x loads) and write a JSON summary.
usage: analyse2.py <out dir of batch 2> <out dir of batch 1> <json>"""
import sys, glob, re, os, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
O2, O1, JS = sys.argv[1:4]
M = ['B1', 'B2', 'HeurMC', 'Proposed', 'Hybrid']
COLS = ['delay', 'pktdelay', 'pdr', 'thr', 'fair', 'loadvar', 'handover', 'm_delay', 'm_pdr', 'm_thr']
def per_seed(d, m):
    S = {}
    for f in glob.glob(f'{d}/{m}_s*_e*.csv'):
        s = int(re.findall(r'_s(\d+)_e', f)[0]); S.setdefault(s, []).append(pd.read_csv(f).mean())
    return pd.DataFrame({s: pd.DataFrame(v).mean() for s, v in sorted(S.items())}).T if S else None
dirs = {('A10', '1.0'): f'{O1}/L1.0', ('A10', '0.3'): f'{O1}/L0.3', ('A10', '0.05'): f'{O2}/A10/L0.05'}
for c in ['A5', 'A20', 'B10', 'C10']:
    for L in ['1.0', '0.3', '0.05']: dirs[(c, L)] = f'{O2}/{c}/L{L}'
out = {}
for (c, L), d in dirs.items():
    D = {m: per_seed(d, m) for m in M}
    if any(v is None for v in D.values()): print('missing', c, L); continue
    key = f'{c}_L{L}'; out[key] = {'n_episodes': int(len(glob.glob(f'{d}/Hybrid_s1_e*.csv')))}
    for m in M:
        out[key][m] = {k: [float(D[m][k].mean()), float(D[m][k].std(ddof=1))] for k in COLS}
    out[key]['p_vs_Hybrid'] = {}
    for m in M[:-1]:
        out[key]['p_vs_Hybrid'][m] = {k: (1.0 if np.allclose(D['Hybrid'][k], D[m][k]) else
                                          float(wilcoxon(D['Hybrid'][k], D[m][k], method='exact').pvalue))
                                      for k in ['delay', 'pktdelay', 'pdr', 'thr', 'fair', 'loadvar']}
    out[key]['p_LA_vs_B1'] = {k: float(wilcoxon(D['Proposed'][k], D['B1'][k], method='exact').pvalue)
                              for k in ['delay', 'pktdelay', 'pdr', 'thr']}
    r = out[key]
    print(f"\n== {key} (eps/seed {r['n_episodes']})")
    for m in M:
        x = r[m]
        print(f"{m:9s} delay {x['delay'][0]:7.0f} pkt {x['pktdelay'][0]:6.0f} pdr {x['pdr'][0]:.3f} thr {x['thr'][0]:6.1f} "
              f"fair {x['fair'][0]:.2f} lv {x['loadvar'][0]:6.1f} | model delay {x['m_delay'][0]:6.1f} pdr {x['m_pdr'][0]:.3f} thr {x['m_thr'][0]:6.1f}")
    print('  p(Hybrid vs):', {m: {k: round(v, 4) for k, v in d.items() if k in ('delay', 'thr', 'pdr')} for m, d in r['p_vs_Hybrid'].items()})
json.dump(out, open(JS, 'w'), indent=1)
