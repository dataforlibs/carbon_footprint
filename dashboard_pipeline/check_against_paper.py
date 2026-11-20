"""
check_against_paper.py  --  step 3.  Compares every condition (docs 0, months 1) with Tables S1/S2 of the supplement,
using the table's own definition (windows from each country's first publication; CV averaged over all countries
with a fit).  Prints OK / DIFF for each number, so every difference is listed.

    python check_against_paper.py --data dashboard_data
"""
import argparse, json
ORDER = ['A-3M', 'A-6M', 'A-12M', 'B-3M', 'B-6M', 'B-12M']
PAPER = {  # Table S2 (no padding) and Table S1 (padded): windows, median x0, x0<=1 %, entities with folds, folds, x0 CV, k CV
 0:   dict(windows=[4504, 2302, 1163, 4570, 2361, 1192], median=[.850, .857, .863, .848, .846, .833],
           le1=[80.7, 78.3, 73.2, 82.3, 82.4, 84.2], ewf=[113, 109, 86, 114, 112, 90],
           folds=[1382, 552, 159, 1440, 630, 214], x0cv=[.386, .182, .186, .176, .174, .166],
           kcv=[2.255, 1.894, 1.385, 2.377, 2.112, 1.710]),
 100: dict(windows=[4480, 2259, 1088, 4570, 2361, 1192], median=[.959, .977, .987, .919, .918, .912],
           le1=[82.3, 80.7, 79.6, 83.2, 83.2, 84.8], ewf=[50, 0, 0, 86, 79, 38],
           folds=[86, 0, 0, 704, 272, 53], x0cv=[.074, .057, .037, .105, .103, .097],
           kcv=[2.369, 2.100, 1.639, 2.371, 2.132, 1.710])}
TOL = dict(windows=0, median=.0006, le1=.06, ewf=0, folds=0, x0cv=.0006, kcv=.0006)
TEXT = {'paris_back_unpadded_A6': 139, 'covid_back_unpadded_A6': 135, 'paris_fwd_unpadded_A6': 14, 'covid_fwd_unpadded_A6': 44}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='dashboard_data'); a = ap.parse_args()
    S = json.load(open(f'{a.data}/scenario_summaries.json')); bad = 0; n = 0
    for pad, P in PAPER.items():
        print(f'\n== padding {pad} ==')
        for i, lab in enumerate(ORDER):
            ap_, W = lab[0], lab[2:-1]
            s = S.get(f'p{pad}_{ap_}_{W}_d0_m1')
            if not s: print(f'{lab}: not built'); continue
            pd_ = s['paper_definition']
            got = dict(windows=pd_['windows'], median=pd_['median_x0'], le1=pd_['pct_x0_le1'], ewf=s['entities_with_folds'],
                       folds=s['folds']['back']['n'], x0cv=pd_['x0_cv'], kcv=pd_['k_cv'])
            diffs = []
            for k, v in got.items():
                n += 1
                if abs(v - P[k][i]) > TOL[k]:
                    bad += 1; diffs.append(f'{k}: paper {P[k][i]} vs data {v:.3f}')
            print(f'{lab:6s}', 'OK' if not diffs else 'DIFF  ' + '; '.join(diffs))
    s = S.get('p0_A_6_d0_m1')
    if s:
        print('\n== Section 4.2.5 (unpadded A-6M) ==')
        f = s['folds']
        for name, got, paper in [('Paris backward', f['back']['paris']['n'], 139), ('COVID backward', f['back']['covid']['n'], 135),
                                 ('Paris forward', f['fwd']['paris']['n'], 14), ('COVID forward', f['fwd']['covid']['n'], 44)]:
            print(f'{name:15s} paper {paper:4d}  data {got:4d}  ', 'OK' if got == paper else 'DIFF')
    print(f'\n{n-bad}/{n} table numbers match')
