"""
run_grid.py  --  step 1.  Fits the 12 base conditions (2 paddings x 2 approaches x 3 window sizes).

    python run_grid.py --csv pivoted_data_states_monthly.csv --out base_fits --jobs 4

Resumable: conditions whose .pkl already exists are skipped.
Rough time: the 3M and monthly (B) conditions are the slow ones; use --jobs = number of cores.
"""
import argparse, os, itertools, time
from concurrent.futures import ProcessPoolExecutor
from sens_engine import fit_condition, BASE_SCRIPT


def job(a):
    csv, pad, ap, W, out, base = a
    fn = os.path.join(out, f'pad{pad}_{ap}_{W}M.pkl')
    if os.path.exists(fn):
        return fn, 0.0, 'skipped'
    t = time.time(); fit_condition(csv, pad, ap, W, out, base)
    return fn, time.time() - t, 'done'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='pivoted_data_states_monthly.csv')
    ap.add_argument('--out', default='base_fits')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--base', default=BASE_SCRIPT)
    ap.add_argument('--only', default='', help='e.g. "0_A_12" to run a single condition: pad_approach_window')
    a = ap.parse_args()
    conds = list(itertools.product([0, 100], ['A', 'B'], [3, 6, 12]))
    if a.only:
        p, x, w = a.only.split('_'); conds = [(int(p), x, int(w))]
    os.makedirs(a.out, exist_ok=True)
    tasks = [(a.csv, p, x, w, a.out, a.base) for p, x, w in conds]
    with ProcessPoolExecutor(a.jobs) as ex:
        for fn, sec, st in ex.map(job, tasks):
            print(f'{st:8s} {fn}  {sec:6.0f}s', flush=True)
