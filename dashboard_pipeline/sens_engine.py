"""
sens_engine.py  --  fits ONE base condition (padding x approach x window size) and saves every window fit.

Why only 12 fits are needed: the fit at window i always uses all data up to window i, so the
"minimum documents" and "minimum months" rules only decide WHICH windows are kept. Those rules
are applied later, in build_dashboard_data.py, without refitting.

It reuses the fitting, phase and grouping code of your May script (imported, not copied), so the
fits are identical to the paper's.
"""
import importlib.util, pickle, os, sys
import numpy as np, pandas as pd

BASE_SCRIPT = 'sensitivity_analysis_comparison_2_1_2_1_with_forward.py'


def load_base(path=BASE_SCRIPT):
    spec = importlib.util.spec_from_file_location('may_base', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fit5(m, x, y_cum):
    """Same fit as the May script's fit_sigmoid (same numbers), but also returns A_norm
    (raw fitted amplitude in normalised-y space), as your June scripts did."""
    from scipy.optimize import curve_fit
    try:
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = y_cum / (y_cum.max() + 1e-10)
        p, _ = curve_fit(m.sigmoid, x_norm, y_norm,
                         p0=[0.5, 4.0 / max(len(x), 1), 1.0],
                         bounds=([-np.inf, 0, 0], np.inf), maxfev=10000)
        x0n = min(p[0], m.X0_CAP)
        span = x.max() - x.min()
        return (x0n, x0n * span + x.min(), p[1] / (span + 1e-10), p[2] * y_cum.max(), p[2]), True
    except Exception:
        return (np.nan,) * 5, False


def prepare(m, csv, pad):
    """Load, consolidate USA/UK/Turkey and pad exactly as the May script does."""
    m.PAD_MONTHS = pad
    m.TOTAL_MONTHS = pad + m.DATA_MONTHS
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= '2023-07-31'].copy().set_index('date')
    df = df.resample('MS').sum().reset_index()
    df = m.group_entities(df)
    df = m.standardise_monthly_series(df)
    return df


def _rec(p, wpos, wi, cum, sy, x0y):
    x0, _, k, A, An = p
    return dict(pos=wpos, wi=int(wi), cum=float(cum), sy=float(sy), x0=float(x0), k=float(k),
                A=float(A), An=float(An), x0y=float(x0y))


def fit_condition(csv, pad, approach, W, outdir, base=BASE_SCRIPT):
    m = load_base(base)
    df = prepare(m, csv, pad)
    countries = [c for c in df.columns if c != 'date']
    dates = pd.to_datetime(df['date'])
    out = {'pad': pad, 'approach': approach, 'W': W, 'countries': {}}
    first_pub = {}
    for c in countries:
        nz = np.nonzero(df[c].values)[0]
        if len(nz):
            d = dates.iloc[nz[0]]
            first_pub[c] = d.year + (d.month - 1) / 12
    out['first_pub_year'] = first_pub

    df_agg = m.aggregate_to_window(df, W)
    n_total = len(df_agg); n_data = n_total - pad
    agg_dates = pd.to_datetime(df_agg['date'])
    delta = 1.0 / (n_data - 1)
    out['delta'] = delta; out['n_data_windows'] = n_data
    out['window_years'] = [float(agg_dates.iloc[pad + i].year + agg_dates.iloc[pad + i].dayofyear / 365.25)
                           for i in range(n_data)]

    if approach == 'B':
        n_m = len(df)
        snap_idx = []
        for wi in range(pad + 1, n_total):
            end = agg_dates.iloc[wi + 1] - pd.Timedelta(days=1) if wi + 1 < n_total else dates.max()
            mask = dates <= end
            if mask.any():
                snap_idx.append(int(mask[mask].index[-1]))
        snap_idx = sorted(set(snap_idx))

    for c in countries:
        pubs = (df_agg[c].values if approach == 'A' else df[c].values)
        if df[c].values.sum() == 0:
            continue
        rows = []
        if approach == 'A':
            for wi in range(pad + 1, n_total):
                cum = np.cumsum(pubs[:wi + 1]); t = np.arange(wi + 1)
                p, ok = fit5(m, t, cum)
                if not ok: continue
                pos = wi - pad                       # 1 .. n_data-1
                cur = pos / (n_data - 1)
                sd = agg_dates.iloc[wi]; sy = sd.year + sd.dayofyear / 365.25
                span = (sd - agg_dates.iloc[0]).days; fr = p[1] / max(wi, 1)
                xd = agg_dates.iloc[0] + pd.Timedelta(days=max(0, int(fr * span)))
                r = _rec(p, pos, wi, cum[-1], sy, xd.year + xd.dayofyear / 365.25)
                r['ph'] = m.classify_phase(p[0], cur, p[3], cum[-1], delta)
                rows.append(r)
        else:
            for i, wi in enumerate(snap_idx):
                cum = np.cumsum(pubs[:wi + 1]); t = np.arange(wi + 1)
                p, ok = fit5(m, t, cum)
                if not ok: continue
                pos = 1 + i
                cur = pos / (n_data - 1)
                sd = dates.iloc[wi]; sy = sd.year + sd.dayofyear / 365.25
                span = (sd - dates.iloc[0]).days; fr = p[1] / max(wi, 1)
                xd = dates.iloc[0] + pd.Timedelta(days=max(0, int(fr * span)))
                r = _rec(p, pos, wi, cum[-1], sy, xd.year + xd.dayofyear / 365.25)
                r['ph'] = m.classify_phase(p[0], cur, p[3], cum[-1], delta)
                rows.append(r)
        out['countries'][c] = rows
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f'pad{pad}_{approach}_{W}M.pkl')
    pickle.dump(out, open(fn, 'wb'))
    return fn
