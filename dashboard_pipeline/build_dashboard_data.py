"""
build_dashboard_data.py  --  step 2.  Turns the 12 base fits into everything the dashboard reads.

    python build_dashboard_data.py --fits base_fits --out dashboard_data

Writes
  dashboard_data/manifest.json                 scenario list, factor values, data fingerprint
  dashboard_data/scenario_summaries.json       one small summary per scenario (all of them)
  dashboard_data/sens_grid/pad{0|100}_docs{0-5}_months{1|6|12|24}/{aggregated_fit|monthly_snapshot}/{3M|6M|12M}/country_results.json
        in the SAME format as your lifepaths_output2 files (window_params with x0_norm, x0_year, k, A,
        A_norm, phase, snapshot_year, wi, cum_pubs), so the existing lifepath plot code draws them unchanged.

Rules applied here without refitting (a fit never depends on where the analysis starts):
  min documents : keep a window only if the country's cumulative publications >= docs  (0 = keep all, as in May)
  min months    : keep a window only if >= months have passed since the country's first publication
                  (1 = no rule). Measured at the window's snapshot date.
Folds are re-detected on the kept windows (x0 <= 1 in both windows, step > delta = 1/(n_data-1)).
"""
import argparse, glob, json, os, pickle, datetime, hashlib
import numpy as np

SAT = {'Deactivated', 'Deceleration'}
PARIS, COVID = (2015, 2016), (2019, 2021)
DOCS = [0, 1, 2, 3, 4, 5]
MONTHS = [1, 6, 12, 24]
YEARS = list(range(2007, 2024))
APPR = {'A': 'aggregated_fit', 'B': 'monthly_snapshot'}


def perm_test(years, lo, hi, n_perm=10000):
    fy = np.asarray(years, float)
    if len(fy) == 0:
        return 0, None, 0.0
    rng = np.random.RandomState(42)
    width = hi - lo
    yr_min, yr_max = int(fy.min()), int(fy.max())
    obs = int(np.sum((fy >= lo) & (fy <= hi)))
    hi_rand = max(yr_max - width, yr_min + 1)
    starts = rng.randint(yr_min, hi_rand + 1, n_perm)
    perm = np.array([np.sum((fy >= s) & (fy <= s + width)) for s in starts])
    return obs, float(np.mean(perm >= obs)), float(perm.mean())


def keep(rows, docs, months, first_year):
    out = []
    for r in rows:
        if docs > 0 and r['cum'] < docs:
            continue
        if months > 1:
            if first_year is None or (r['sy'] - first_year) * 12 < months:
                continue
        out.append(r)
    return out


def folds(rows, delta):
    back, fwd = [], []
    for a, b in zip(rows[:-1], rows[1:]):
        if a['x0'] <= 1.0 and b['x0'] <= 1.0:
            if b['x0'] < a['x0'] - delta: back.append((b['sy'], b['ph']))
            elif b['x0'] > a['x0'] + delta: fwd.append((b['sy'], b['ph']))
    return back, fwd


def country_record(rows, total_pubs, delta, n_fit_points):
    back, fwd = folds(rows, delta)
    rec = {
        'window_params': [{'x0_norm': r['x0'], 'k': r['k'], 'A': r['A'], 'A_norm': r['An'], 'phase': r['ph'],
                           'snapshot_year': r['sy'], 'x0_year': r['x0y'], 'wi': r['wi'], 'cum_pubs': r['cum']}
                          for r in rows],
        'phases': [r['ph'] for r in rows],
        'final_phase': rows[-1]['ph'] if rows else None,
        'time_folds': len(back), 'fold_years': [y for y, _ in back],
        'saturation_fold_years': [y for y, p in back if p in SAT],
        'time_folds_forward': len(fwd), 'forward_fold_years': [y for y, _ in fwd],
        'fwd_sat_fold_years': [y for y, p in fwd if p in SAT],
        'final_x0_year': rows[-1]['x0y'] if rows else None,
        'total_pubs': total_pubs, 'n_fit_points': n_fit_points,
    }
    if rows:
        x0 = np.array([r['x0'] for r in rows]); k = np.array([r['k'] for r in rows])
        rec['median_x0'] = float(np.median(x0)); rec['pct_x0_le1'] = float(100 * np.mean(x0 <= 1.0))
        rec['param_stability'] = {'x0_cv': float(x0.std() / (x0.mean() + 1e-10)),
                                  'k_cv': float(k.std() / (k.mean() + 1e-10))}
    return rec


def by_year(years):
    a = np.asarray(years, float)
    return [int(((a >= y) & (a < y + 1)).sum()) for y in YEARS]


def summarise(res, base, paper_rows, kept):
    allx = [w['x0_norm'] for r in res.values() for w in r['window_params']]
    s = {'countries_analyzed': sum(1 for r in res.values() if r['window_params']),
         'windows': len(allx),
         'median_x0': float(np.median(allx)) if allx else None,
         'pct_x0_le1': float(100 * np.mean(np.array(allx) <= 1)) if allx else None,
         'x0_cv': float(np.mean([r['param_stability']['x0_cv'] for r in res.values() if 'param_stability' in r])) if allx else None,
         'k_cv': float(np.mean([r['param_stability']['k_cv'] for r in res.values() if 'param_stability' in r])) if allx else None,
         'entities_with_folds': sum(1 for r in res.values() if r['time_folds'] > 0)}
    cols = {'back': 'fold_years', 'back_sat': 'saturation_fold_years', 'fwd': 'forward_fold_years', 'fwd_sat': 'fwd_sat_fold_years'}
    allf = {k: [y for r in res.values() for y in r[c]] for k, c in cols.items()}
    allf['total'] = allf['back'] + allf['fwd']
    allf['total_sat'] = allf['back_sat'] + allf['fwd_sat']
    s['folds'] = {}
    for k, ys in allf.items():
        P = perm_test(ys, *PARIS); C = perm_test(ys, *COVID)
        s['folds'][k] = {'n': len(ys), 'by_year': by_year(ys),
                         'paris': {'n': P[0], 'p': P[1]}, 'covid': {'n': C[0], 'p': C[1]}}
    fin = [r['final_phase'] for r in res.values() if r['final_phase']]
    s['final_phase_pct'] = {p: round(100 * fin.count(p) / len(fin), 1) for p in ['Acceleration', 'Inflection', 'Deceleration', 'Deactivated']} if fin else {}
    s['final_inflection_years'] = [round(r['final_x0_year'], 2) for r in res.values() if r['final_x0_year'] is not None]
    s['window_years'] = base['window_years']
    cnt = [0] * base['n_data_windows']          # countries with a fit at each window (index = window position)
    for rows in kept.values():
        for r in rows:
            cnt[r['pos']] += 1
    s['countries_by_window'] = cnt
    # definition used in the paper's tables S1/S2: windows from the country's first publication
    px = [w for rows in paper_rows.values() for w in rows]
    if px:
        x = np.array([w['x0'] for w in px])
        cvx = [np.std([w['x0'] for w in rows]) / (np.mean([w['x0'] for w in rows]) + 1e-10) for rows in paper_rows.values()]
        cvk = [np.std([w['k'] for w in rows]) / (np.mean([w['k'] for w in rows]) + 1e-10) for rows in paper_rows.values()]
        s['paper_definition'] = {'windows': len(x), 'median_x0': float(np.median(x)), 'pct_x0_le1': float(100 * np.mean(x <= 1)),
                                 'x0_cv': float(np.mean(cvx)), 'k_cv': float(np.mean(cvk))}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fits', default='base_fits'); ap.add_argument('--out', default='dashboard_data')
    ap.add_argument('--csv', default='pivoted_data_states_monthly.csv')
    ap.add_argument('--country-files', choices=['core', 'all'], default='core',
                    help='core = 12 base conditions + the docs/months sweeps from the two anchor conditions; all = every scenario')
    a = ap.parse_args()
    summaries, manifest = {}, {'scenarios': [], 'built': datetime.datetime.now().isoformat(timespec='seconds'),
                               'data_file': a.csv, 'data_bytes': os.path.getsize(a.csv) if os.path.exists(a.csv) else None,
                               'data_md5': hashlib.md5(open(a.csv, 'rb').read()).hexdigest() if os.path.exists(a.csv) else None}
    for fn in sorted(glob.glob(os.path.join(a.fits, 'pad*_*M.pkl'))):
        base = pickle.load(open(fn, 'rb')); pad, apr, W = base['pad'], base['approach'], base['W']
        delta = base['delta']; fy = base['first_pub_year']
        total = {c: (max(r['cum'] for r in rows) if rows else 0) for c, rows in base['countries'].items()}
        for d in DOCS:
            for m in MONTHS:
                core = (d == 0 and m == 1) or (pad in (0, 100) and apr == 'A' and W == 6 and (m == 1 or d == 0))
                key = f'p{pad}_{apr}_{W}_d{d}_m{m}'
                kept = {c: keep(rows, d, m, fy.get(c)) for c, rows in base['countries'].items()}
                res = {c: country_record(kept[c], total[c], delta, base['n_data_windows']) for c in kept}
                paper_rows = {c: [r for r in keep(rows, d, m, fy.get(c)) if r['cum'] >= 1] for c, rows in base['countries'].items()}
                paper_rows = {c: v for c, v in paper_rows.items() if v}
                summaries[key] = summarise(res, base, paper_rows, kept)
                summaries[key]['params'] = {'padding': pad, 'approach': apr, 'window_months': W, 'min_docs': d, 'min_months': m}
                has_files = a.country_files == 'all' or core
                if has_files:
                    folder = os.path.join(a.out, 'sens_grid', f'pad{pad}_docs{d}_months{m}', APPR[apr], f'{W}M')
                    os.makedirs(folder, exist_ok=True)
                    json.dump(res, open(os.path.join(folder, 'country_results.json'), 'w'), separators=(',', ':'))
                manifest['scenarios'].append({'key': key, 'country_results': has_files and
                                              f'sens_grid/pad{pad}_docs{d}_months{m}/{APPR[apr]}/{W}M/country_results.json'})
        print('built', fn, flush=True)
    manifest['n_entities'] = len(base['countries'])
    manifest['factors'] = {'padding': [0, 100], 'approach': ['A', 'B'], 'window_months': [3, 6, 12],
                           'min_docs': DOCS, 'min_months': MONTHS}
    os.makedirs(a.out, exist_ok=True)
    json.dump(manifest, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
    json.dump(summaries, open(os.path.join(a.out, 'scenario_summaries.json'), 'w'), separators=(',', ':'))
    print('scenarios:', len(summaries))


if __name__ == '__main__':
    main()
