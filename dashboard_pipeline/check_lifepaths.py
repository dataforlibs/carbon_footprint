"""
check_lifepaths.py  --  draws lifepaths from a country_results.json using the SAME rules as your fixed dashboard page,
so you can confirm the generated data plots correctly before building the page.

    python check_lifepaths.py dashboard_data/sens_grid/pad0_docs0_months1/aggregated_fit/6M/country_results.json \
           --countries France USA India Brazil --out lifepath_check.png

Rules copied from sensitivity_dashboard2_7_2_2.html:
  x = x0_year (projected inflection date)      y = ln(k), axis REVERSED (the June fix)
  skip windows with x0_norm >= 1.19 (capped), k > 3000, or A / total_pubs > 50
  colours: Acceleration green, Inflection blue, Deceleration red, Deactivated black
  backward fold segment = solid red thick line, forward = dashed orange (step > 1/(n-1) with x0 <= 1 in both)
"""
import argparse, json, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

X0_CAP, K_MAX, A_RATIO_MAX = 1.19, 3000, 50
COL = {'Acceleration': 'green', 'Inflection': 'blue', 'Deceleration': 'red', 'Deactivated': 'black'}


def points(rec):
    wp = rec['window_params']; delta = 1 / (len(wp) - 1) if len(wp) > 1 else 1
    pts = []
    for i, p in enumerate(wp):
        k = p['k']; x0 = min(p['x0_norm'], 1.3)
        if not (k and k > 0) or x0 >= X0_CAP or k > K_MAX: continue
        if rec['total_pubs'] and p['A'] / rec['total_pubs'] > A_RATIO_MAX: continue
        fold = 'none'
        if i:
            a, b = min(wp[i - 1]['x0_norm'], 1.3), x0
            if a <= 1 and b <= 1:
                fold = 'back' if b < a - delta else 'fwd' if b > a + delta else 'none'
        pts.append((p['x0_year'], math.log(k), p['phase'], fold))
    return pts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--countries', nargs='+', default=['France', 'USA', 'India', 'Brazil'])
    ap.add_argument('--out', default='lifepath_check.png'); a = ap.parse_args()
    data = json.load(open(a.file)); n = len(a.countries)
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for ax, c in zip(axs[0], a.countries):
        rec = data.get(c)
        if not rec or not rec['window_params']:
            ax.set_title(f'{c}: no fit'); continue
        pts = points(rec)
        for (x0, y0, _, _), (x1, y1, ph, fold) in zip(pts[:-1], pts[1:]):
            ax.plot([x0, x1], [y0, y1], color='#888' if fold == 'none' else ('red' if fold == 'back' else 'orange'),
                    lw=0.8 if fold == 'none' else 2.4, ls='--' if fold == 'fwd' else '-', alpha=.8)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=[COL[p[2]] for p in pts], s=22, zorder=3)
        ax.invert_yaxis()
        ax.set_xlabel('Projected inflection date (x0_year)'); ax.set_ylabel('ln(k)  (axis reversed)')
        ax.set_title(f"{c}  ·  {rec['final_phase']}  ·  folds {rec['time_folds']}b/{rec['time_folds_forward']}f", fontsize=10)
        ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print('saved', a.out)


if __name__ == '__main__':
    main()
