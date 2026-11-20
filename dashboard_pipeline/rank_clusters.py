"""
rank_clusters.py -- which scenarios show the strongest clustering of folds around Paris (2015-16) and COVID (2019-21)?

    python3 rank_clusters.py --data dashboard_data --top 15

Reads scenario_summaries.json (no refitting).  For every scenario and fold type (backward, forward, total; all or
saturation-phase) it takes the permutation p-value already stored for each event window, then reports:
  1. the raw ranking (smallest p first),
  2. how many scenarios are 'significant' at 0.05 -- and how many you would expect by chance alone,
  3. p-values after Benjamini-Hochberg correction across ALL tests,
  4. a placebo: for the same scenario, how unusual are Paris/COVID compared with every other window of the same width
     placed anywhere in 2007-2023?  (If many ordinary years look as 'clustered' as Paris, the result is not special.)
Only the correction-adjusted numbers (3) and the placebo (4) should be used to claim clustering.
"""
import argparse, json
import numpy as np

EVENTS = {'paris': (2015, 2016), 'covid': (2019, 2021)}
KINDS = ['back', 'fwd', 'total', 'back_sat', 'fwd_sat']
YEARS = list(range(2007, 2024))


def placebo_rank(by_year, lo, hi):
    """Share of same-width windows (anywhere in the data) that hold at least as many folds as [lo, hi]."""
    w = hi - lo
    counts = [sum(by_year[i] for i in range(s - 2007, s - 2007 + w + 1)) for s in range(2007, 2023 - w + 1)]
    obs = sum(by_year[lo - 2007: hi - 2007 + 1])
    return obs, float(np.mean([c >= obs for c in counts]))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n)
    run = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        run = min(run, p[i] * n / rank); adj[i] = run
    return adj


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='dashboard_data'); ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()
    S = json.load(open(f'{a.data}/scenario_summaries.json'))
    rows = []
    for key, s in S.items():
        for kind in KINDS:
            f = s['folds'][kind]
            if f['n'] < 20: continue                       # too few folds to test
            for ev, (lo, hi) in EVENTS.items():
                p = f[ev]['p']
                if p is None: continue
                obs, pp = placebo_rank(f['by_year'], lo, hi)
                rows.append((key, kind, ev, f[ev]['n'], f['n'], p, pp))
    P = np.array([r[5] for r in rows]); adj = bh(P)
    order = np.argsort(P)
    print(f'{len(rows)} tests ({len(S)} scenarios x fold types x 2 events, folds >= 20)')
    print(f'significant at p<0.05: {(P < 0.05).sum()}   expected by chance if nothing real: {0.05 * len(P):.0f}')
    print(f'significant after Benjamini-Hochberg (q<0.05): {(adj < 0.05).sum()}\n')
    print(f'{"scenario":22s}{"folds":9s}{"event":7s}{"n in window":>12s}{"of":>6s}{"perm p":>8s}{"BH q":>7s}{"placebo":>9s}')
    for i in order[:a.top]:
        k, kind, ev, n, N, p, pp = rows[i]
        print(f'{k:22s}{kind:9s}{ev:7s}{n:12d}{N:6d}{p:8.3f}{adj[i]:7.3f}{pp:9.2f}')
    print('\nplacebo = share of same-width windows elsewhere in 2007-2023 with at least as many folds (small = Paris/COVID stands out).')
    for ev in EVENTS:
        sub = [(r, adj[i]) for i, r in enumerate(rows) if r[2] == ev]
        print(f'{ev}: tests {len(sub)}, raw p<0.05: {sum(r[5] < 0.05 for r, _ in sub)}, BH q<0.05: {sum(q < 0.05 for _, q in sub)}')
