"""
sensitivity_analysis_comparison.py

Compares two approaches to window-size sensitivity analysis:

  APPROACH A — Aggregated:
    Publications are summed into 3/6/12-month buckets.
    The sigmoid is fitted on ~67/34/17 aggregated data points.
    Fit quality varies with window size (fewer points at coarser resolution).

  APPROACH B — Snapshot (monthly fit):
    Publications are aggregated to monthly resolution once.
    The sigmoid is always fitted on all available monthly data (up to 199 points).
    Phase is recorded at 3/6/12-month snapshot intervals.
    Fit quality is constant across window conditions; only snapshot frequency varies.

The comparison isolates whether observed differences across window sizes reflect
genuine resolution effects (B) or confounded fit-quality effects (A).
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
DEACTIVATION_MARGIN = 1.05
X0_CAP              = 1.3
MIN_FIT_POINTS      = 1         # minimum windows before fitting (1 = start from 2nd window)
PARIS_LO, PARIS_HI  = 2015, 2016
COVID_LO,  COVID_HI  = 2019, 2021
N_PERM               = 10_000
SATURATION_PHASES    = {'Deactivated', 'Deceleration'}
PAD_MONTHS           = 0   # leading zero-months prepended to every series
DATA_MONTHS          = 199   # data portion (trim or zero-pad to this length)
TOTAL_MONTHS         = PAD_MONTHS + DATA_MONTHS   # 299
np.random.seed(42)
# ─────────────────────────────────────────────────────────────────────────────


def sigmoid(x, x0, k, A):
    return A / (1 + np.exp(-k * (x - x0)))


def fit_sigmoid(x, y_cum):
    """
    Fit sigmoid on cumulative data. x0 capped at X0_CAP.
    Returns: (params, success)  params = (x0_norm, x0_orig, k_orig, A_orig)
    """
    try:
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = y_cum / (y_cum.max() + 1e-10)
        params, _ = curve_fit(
            sigmoid, x_norm, y_norm,
            p0=[0.5, 4.0 / max(len(x), 1), 1.0],
            bounds=([-np.inf, 0, 0], np.inf),
            maxfev=10000
        )
        x0_norm     = min(params[0], X0_CAP)
        x0_orig     = x0_norm * (x.max() - x.min()) + x.min()
        k_orig      = params[1] / (x.max() - x.min() + 1e-10)
        A_orig      = params[2] * y_cum.max()
        return (x0_norm, x0_orig, k_orig, A_orig), True
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan), False


def classify_phase(x0_norm, current_norm, A, y_current, delta_norm):
    if y_current > 0 and (A / y_current) < DEACTIVATION_MARGIN:
        return 'Deactivated'
    delta_t = current_norm - x0_norm
    if delta_t > delta_norm:
        return 'Deceleration'
    elif delta_t < -delta_norm:
        return 'Acceleration'
    else:
        return 'Inflection'


def detect_time_fold(window_params, delta_norm):
    """
    Detect backward jumps (folds) AND forward jumps in x0_norm trajectory.
    Only counts events where x0 <= 1.0 in both consecutive windows
    (excludes pre-inflection fitting convergence artifacts).

    Backward fold  — x0 moves earlier (capacity revision downward, field contracting)
    Forward jump   — x0 moves later  (capacity revision upward, field re-accelerating)

    Returns
    -------
    n_folds             : int   — total backward fold count
    fold_indices        : list  — window indices where backward folds occur
    fold_years          : list  — calendar year (float) at each backward fold
    sat_fold_years      : list  — fold years restricted to saturation-phase windows
    n_fwd_jumps         : int   — total forward jump count
    fwd_jump_indices    : list  — window indices where forward jumps occur
    fwd_jump_years      : list  — calendar year (float) at each forward jump
    sat_fwd_jump_years  : list  — forward jump years restricted to saturation-phase windows
    """
    fold_indices       = []
    fold_years         = []
    sat_fold_years     = []
    fwd_jump_indices   = []
    fwd_jump_years     = []
    sat_fwd_jump_years = []

    for i in range(1, len(window_params)):
        w_prev  = window_params[i - 1]
        w_cur   = window_params[i]
        x0_prev = w_prev.get('x0_norm', np.nan)
        x0_cur  = w_cur.get('x0_norm',  np.nan)
        if np.isnan(x0_prev) or np.isnan(x0_cur):
            continue
        if x0_cur <= 1.0 and x0_prev <= 1.0:
            yr    = w_cur.get('snapshot_year')
            phase = w_cur.get('phase')
            # ── Backward fold: inflection moves earlier ──────────────────────
            if x0_cur < x0_prev - delta_norm:
                fold_indices.append(i)
                if yr is not None:
                    fold_years.append(yr)
                    if phase in SATURATION_PHASES:
                        sat_fold_years.append(yr)
            # ── Forward jump: inflection moves later ─────────────────────────
            elif x0_cur > x0_prev + delta_norm:
                fwd_jump_indices.append(i)
                if yr is not None:
                    fwd_jump_years.append(yr)
                    if phase in SATURATION_PHASES:
                        sat_fwd_jump_years.append(yr)

    return (len(fold_indices),    fold_indices,     fold_years,     sat_fold_years,
            len(fwd_jump_indices), fwd_jump_indices, fwd_jump_years, sat_fwd_jump_years)


def perm_test(fold_years, lo, hi, n_perm=N_PERM):
    """
    Permutation test: are fold events significantly clustered in [lo, hi]?

    Randomly shifts the window of width (hi - lo) across the observed year
    range and counts how often a random window captures ≥ observed folds.

    Returns (observed_count, p_value, expected_count_under_null).
    """
    fy = np.array(fold_years)
    if len(fy) == 0:
        return 0, np.nan, 0.0
    width   = hi - lo
    yr_min  = int(fy.min())
    yr_max  = int(fy.max())
    obs     = int(np.sum((fy >= lo) & (fy <= hi)))
    hi_rand = max(yr_max - width, yr_min + 1)
    starts  = np.random.randint(yr_min, hi_rand + 1, n_perm)
    perm    = np.array([int(np.sum((fy >= s) & (fy <= s + width))) for s in starts])
    p       = float(np.mean(perm >= obs))
    exp     = float(perm.mean())
    return obs, round(p, 3), round(exp, 1)


def detect_inflection_clusters(inflection_years, bandwidth=2.0, min_count=3):
    """
    Find temporal clusters in the distribution of inflection points (x0 in
    calendar years) using Gaussian KDE peak detection.

    Parameters
    ----------
    inflection_years : list of float — x0 converted to calendar years
    bandwidth        : float — KDE bandwidth in years
    min_count        : int   — minimum countries near a peak to report it

    Returns
    -------
    List of dicts sorted by density: {year, count, pct, density}
    """
    iy = np.array([y for y in inflection_years if not np.isnan(y)])
    if len(iy) < min_count:
        return []
    bw = max(bandwidth / (iy.std() + 1e-9), 0.05)
    kde     = gaussian_kde(iy, bw_method=bw)
    grid    = np.linspace(iy.min() - 1, iy.max() + 1, 600)
    density = kde(grid)
    peaks, _ = find_peaks(density, height=density.max() * 0.15, distance=15)
    clusters = []
    for p in peaks:
        year  = grid[p]
        count = int(np.sum(np.abs(iy - year) <= bandwidth))
        if count >= min_count:
            clusters.append({
                'year':    round(year, 1),
                'count':   count,
                'pct':     round(100 * count / len(iy), 1),
                'density': float(density[p]),
            })
    return sorted(clusters, key=lambda c: -c['density'])


# ── REQUIRED: do not remove — called in __main__ after loading ────────────────
def group_entities(df):
    """Aggregate sub-national and duplicate entries into canonical countries."""
    df = df.copy()
    usa_cols = [c for c in df.columns if 'USA' in c]
    df['USA'] = df[usa_cols].sum(axis=1)
    df.drop(columns=usa_cols, inplace=True)

    uk_cols = [c for c in ['England', 'Scotland', 'North Ireland', 'Wales'] if c in df.columns]
    if uk_cols:
        df['UK'] = df[uk_cols].sum(axis=1)
        df.drop(columns=uk_cols, inplace=True)

    if 'Turkey' in df.columns and 'Turkiye' in df.columns:
        df['Turkey'] += df['Turkiye']
        df.drop(columns=['Turkiye'], inplace=True)
    elif 'Turkiye' in df.columns:
        df.rename(columns={'Turkiye': 'Turkey'}, inplace=True)

    if 'Peoples R China' in df.columns:
        df.rename(columns={'Peoples R China': 'China'}, inplace=True)

    n = len([c for c in df.columns if c != 'date'])
    print(f"  After grouping: {n} entities")
    return df
# ─────────────────────────────────────────────────────────────────────────────


def standardise_monthly_series(df_monthly):
    """
    Standardise all country series to exactly TOTAL_MONTHS = 299 months:
      • Data portion  : trim or zero-pad the actual observations to DATA_MONTHS (199)
      • Leading zeros : prepend PAD_MONTHS (100) synthetic zero-months before the data
    
    All countries share the same date index (they are columns of the same dataframe),
    so a single trim/pad operation on the dataframe rows applies uniformly to every
    entity.  The synthetic leading dates keep the date column valid so that
    aggregate_to_window's resample() continues to work correctly.
    
    Returns a dataframe with exactly TOTAL_MONTHS rows and the same columns.
    """
    country_cols = [c for c in df_monthly.columns if c != 'date']
    data_dates   = pd.to_datetime(df_monthly['date'])
    n_actual     = len(df_monthly)

    # ── 1. Bring the data portion to exactly DATA_MONTHS rows ─────────────────
    if n_actual >= DATA_MONTHS:
        df_data = df_monthly.iloc[:DATA_MONTHS].copy()
    else:
        # Zero-pad trailing months so every entity has the same data length
        n_trail = DATA_MONTHS - n_actual
        last_date   = data_dates.iloc[-1]
        trail_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_trail, freq='MS'
        )
        df_trail = pd.DataFrame({'date': trail_dates})
        for col in country_cols:
            df_trail[col] = 0.0
        df_data = pd.concat([df_monthly, df_trail], ignore_index=True)

    # ── 2. Prepend PAD_MONTHS synthetic zero-months ───────────────────────────
    first_date = pd.to_datetime(df_data['date'].iloc[0])
    pad_dates  = pd.date_range(
        end=first_date - pd.DateOffset(months=1),
        periods=PAD_MONTHS, freq='MS'
    )
    df_pad = pd.DataFrame({'date': pad_dates})
    for col in country_cols:
        df_pad[col] = 0.0

    df_std = pd.concat([df_pad, df_data], ignore_index=True)
    df_std = df_std.sort_values('date').reset_index(drop=True)

    assert len(df_std) == TOTAL_MONTHS, (
        f"Series length mismatch: expected {TOTAL_MONTHS}, got {len(df_std)}")

    print(f"  Standardised: {TOTAL_MONTHS} months total "
          f"({PAD_MONTHS} zero-padding + {DATA_MONTHS} data)  "
          f"[{df_std['date'].iloc[0].date()} → {df_std['date'].iloc[-1].date()}]")
    return df_std






# ─────────────────────────────────────────────────────────────────────────────
# APPROACH A: Aggregated fitting
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_window(df, window_months):
    """
    Aggregates only the DATA portion of df (rows PAD_MONTHS onward) to
    window_months buckets, then prepends exactly PAD_MONTHS synthetic
    zero-window rows — so every resolution has the same 100-window leading
    pad regardless of window size.

    Structure returned:
        rows 0..PAD_MONTHS-1  : synthetic zero-windows (pre-data padding)
        rows PAD_MONTHS..end  : aggregated data windows  (n_data_windows rows)
    Total: PAD_MONTHS + n_data_windows  (varies by resolution; data part ~34/67/17)
    """
    freq = {'3': 'QS', '6': '6MS', '12': 'YS'}[str(window_months)]

    # ── Aggregate only the data portion (skip the PAD_MONTHS leading zeros) ───
    df_data = df.copy().iloc[PAD_MONTHS:].copy()
    df_data['date'] = pd.to_datetime(df_data['date'])
    df_data = df_data.set_index('date')
    df_agg_data = df_data.resample(freq).sum().reset_index()

    country_cols  = [c for c in df_agg_data.columns if c != 'date']
    first_date    = pd.to_datetime(df_agg_data['date'].iloc[0])

    # ── Prepend PAD_MONTHS synthetic zero-window rows ─────────────────────────
    # Dates are spaced by window_months going backward from the data start,
    # so the structure mirrors the PAD_MONTHS/DATA_MONTHS split in Approach B.
    pad_dates = [first_date - pd.DateOffset(months=(PAD_MONTHS - i) * window_months)
                 for i in range(PAD_MONTHS)]
    df_pad = pd.DataFrame({'date': pad_dates})
    for col in country_cols:
        df_pad[col] = 0.0

    df_agg = pd.concat([df_pad, df_agg_data], ignore_index=True)
    df_agg = df_agg.sort_values('date').reset_index(drop=True)
    return df_agg


def run_aggregated(df_monthly, window_months):
    """
    Approach A: aggregate to window_months, fit on aggregated points.
    Series structure: PAD_MONTHS zero-windows + n_data_windows data-windows.
    Fitting loops only over data windows; normalization is over the full
    n_total range so cur_norm is comparable to Approach B.
    """
    df_agg    = aggregate_to_window(df_monthly, window_months)
    countries = [c for c in df_agg.columns if c != 'date']
    n_total   = len(df_agg)                    # PAD_MONTHS + n_data_windows
    n_data    = n_total - PAD_MONTHS           # data windows only
    delta_norm = 1.0 / (n_data - 1)           # normalized over data portion only
    agg_dates  = pd.to_datetime(df_agg['date'])

    print(f"\n  [A] {window_months}M aggregated | "
          f"{n_total} total ({PAD_MONTHS} pad + {n_data} data windows) | "
          f"delta={delta_norm:.4f}")

    results  = {}
    skipped  = 0
    for country in countries:
        pubs       = df_agg[country].values
        total_pubs = pubs.sum()
        if total_pubs == 0:
            skipped += 1
            continue

        window_params, phases = [], []
        # Start from first data window (skip the PAD_MONTHS zero-pad windows)
        for wi in range(PAD_MONTHS + MIN_FIT_POINTS, n_total):
            pubs_sub  = pubs[:wi+1]
            cum_pubs  = np.cumsum(pubs_sub)
            time_pts  = np.arange(len(pubs_sub))
            params, ok = fit_sigmoid(time_pts, cum_pubs)
            if ok:
                x0_norm, x0_orig, k, A = params
                # cur_norm over full range [0, n_total-1] so it matches B
                cur_norm = (wi - PAD_MONTHS) / (n_data - 1)
                phase    = classify_phase(x0_norm, cur_norm, A, cum_pubs[-1], delta_norm)

                # Calendar year at this snapshot date
                snap_date     = agg_dates.iloc[wi]
                snapshot_year = snap_date.year + snap_date.dayofyear / 365.25

                # Calendar year of the inflection point x0_orig
                # x0_orig is in time_pts units (0..wi); interpolate over date range
                start_date = agg_dates.iloc[0]
                span_days  = (snap_date - start_date).days
                frac       = x0_orig / max(wi, 1)
                x0_date    = start_date + pd.Timedelta(days=max(0, int(frac * span_days)))
                x0_year    = x0_date.year + x0_date.dayofyear / 365.25

                window_params.append({
                    'x0_norm': x0_norm, 'k': k, 'A': A, 'phase': phase,
                    'snapshot_year': snapshot_year, 'x0_year': x0_year,
                })
                phases.append(phase)

        if window_params:
            x0v = [w['x0_norm'] for w in window_params]
            kv  = [w['k']       for w in window_params]
            (nf, _, fold_years, sat_fold_years,
             nfj, _, fwd_jump_years, sat_fwd_jump_years) = detect_time_fold(window_params, delta_norm)
            final_x0_year = window_params[-1]['x0_year']
            results[country] = {
                'window_params':              window_params,
                'phases':                     phases,
                'final_phase':                window_params[-1]['phase'],
                'time_folds':                 nf,
                'fold_years':                 fold_years,
                'saturation_fold_years':      sat_fold_years,
                'fwd_jumps':                  nfj,
                'fwd_jump_years':             fwd_jump_years,
                'saturation_fwd_jump_years':  sat_fwd_jump_years,
                'final_x0_year':              final_x0_year,
                'median_x0':                  np.median(x0v),
                'pct_x0_le1':                 100 * np.mean([v <= 1.0 for v in x0v]),
                'param_stability': {
                    'x0_cv': np.std(x0v) / (np.mean(x0v) + 1e-10),
                    'k_cv':  np.std(kv)  / (np.mean(kv)  + 1e-10),
                },
                'total_pubs':    total_pubs,
                'n_fit_points':  n_total,
            }

    print(f"  Fitted: {len(results)} | Skipped: {skipped}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH B: Monthly fit, snapshot at intervals
# ─────────────────────────────────────────────────────────────────────────────

def run_snapshot(df_monthly, snapshot_months):
    """
    Approach B: always fit on monthly data up to snapshot date.
    Phase recorded only at the same window boundaries as Approach A.

    Alignment: snap_idx is derived from A's aggregated window dates, but
    only for DATA windows (wi >= PAD_MONTHS) — the 100 synthetic pad-windows
    in df_agg have no corresponding monthly data and are skipped.

    cur_norm uses the same denominator (n_total - 1) as A so phase
    classifications remain directly comparable.
    """
    countries     = [c for c in df_monthly.columns if c != 'date']
    monthly_dates = pd.to_datetime(df_monthly['date'])
    n_monthly     = len(df_monthly)

    # ── Align with Approach A ─────────────────────────────────────────────────
    df_agg      = aggregate_to_window(df_monthly, snapshot_months)
    n_total     = len(df_agg)                  # PAD_MONTHS + n_data_windows
    n_data      = n_total - PAD_MONTHS
    agg_dates   = pd.to_datetime(df_agg['date'])
    min_windows = MIN_FIT_POINTS

    # Build snap_idx only from DATA windows (skip the PAD_MONTHS synthetic rows)
    snap_idx = []
    for wi in range(PAD_MONTHS + min_windows, n_total):
        if wi + 1 < n_total:
            window_end = agg_dates.iloc[wi + 1] - pd.Timedelta(days=1)
        else:
            window_end = monthly_dates.max()
        mask = monthly_dates <= window_end
        if mask.any():
            snap_idx.append(int(mask[mask].index[-1]))
    snap_idx = sorted(set(snap_idx))
    n_snaps  = len(snap_idx)

    delta_norm = 1.0 / (n_data - 1) if n_data > 1 else 1.0

    assert n_snaps == n_data - min_windows, (
        f"Snapshot count mismatch: B has {n_snaps}, "
        f"A has {n_data - min_windows} data windows")

    print(f"\n  [B] {snapshot_months}M snapshot | {n_monthly} monthly pts | "
          f"{n_snaps} snapshots (aligned with A's {n_data} data windows, "
          f"{PAD_MONTHS} pad skipped) | delta={delta_norm:.4f}")

    results  = {}
    skipped  = 0
    for country in countries:
        pubs       = df_monthly[country].values
        total_pubs = pubs.sum()
        if total_pubs == 0:
            skipped += 1
            continue

        window_params, phases = [], []
        for snap_i, wi in enumerate(snap_idx):
            pubs_sub  = pubs[:wi+1]
            cum_pubs  = np.cumsum(pubs_sub)
            time_pts  = np.arange(len(pubs_sub))
            params, ok = fit_sigmoid(time_pts, cum_pubs)
            if ok:
                x0_norm, x0_orig, k, A = params
                # cur_norm over data range [0, n_data-1], matching A's scale
                cur_norm = (min_windows + snap_i) / (n_data - 1)
                phase    = classify_phase(x0_norm, cur_norm, A, cum_pubs[-1], delta_norm)

                # Calendar year at this snapshot date (monthly index wi)
                snap_date     = monthly_dates.iloc[wi]
                snapshot_year = snap_date.year + snap_date.dayofyear / 365.25

                # Calendar year of inflection point x0_orig (in monthly units 0..wi)
                start_date = monthly_dates.iloc[0]
                span_days  = (snap_date - start_date).days
                frac       = x0_orig / max(wi, 1)
                x0_date    = start_date + pd.Timedelta(days=max(0, int(frac * span_days)))
                x0_year    = x0_date.year + x0_date.dayofyear / 365.25

                window_params.append({
                    'x0_norm': x0_norm, 'k': k, 'A': A, 'phase': phase,
                    'snap_i': snap_i,
                    'snapshot_year': snapshot_year, 'x0_year': x0_year,
                })
                phases.append(phase)

        if window_params:
            x0v = [w['x0_norm'] for w in window_params]
            kv  = [w['k']       for w in window_params]
            (nf, _, fold_years, sat_fold_years,
             nfj, _, fwd_jump_years, sat_fwd_jump_years) = detect_time_fold(window_params, delta_norm)
            final_x0_year = window_params[-1]['x0_year']
            results[country] = {
                'window_params':              window_params,
                'phases':                     phases,
                'final_phase':                window_params[-1]['phase'],
                'time_folds':                 nf,
                'fold_years':                 fold_years,
                'saturation_fold_years':      sat_fold_years,
                'fwd_jumps':                  nfj,
                'fwd_jump_years':             fwd_jump_years,
                'saturation_fwd_jump_years':  sat_fwd_jump_years,
                'final_x0_year':              final_x0_year,
                'median_x0':                  np.median(x0v),
                'pct_x0_le1':                 100 * np.mean([v <= 1.0 for v in x0v]),
                'param_stability': {
                    'x0_cv': np.std(x0v) / (np.mean(x0v) + 1e-10),
                    'k_cv':  np.std(kv)  / (np.mean(kv)  + 1e-10),
                },
                'total_pubs':    total_pubs,
                'n_fit_points':  n_monthly,
            }

    print(f"  Fitted: {len(results)} | Skipped: {skipped}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary and comparison
# ─────────────────────────────────────────────────────────────────────────────

def summarise(results, label):
    if not results:
        return {}
    all_x0, all_phases, final_phases = [], [], []
    total_wins = 0
    all_fold_years          = []
    all_sat_fold_years      = []
    all_fwd_jump_years      = []
    all_sat_fwd_jump_years  = []
    all_final_x0_years      = []
    n_sat_with_folds        = 0
    n_sat_with_fwd_jumps    = 0

    for r in results.values():
        all_x0.extend([w['x0_norm'] for w in r['window_params']])
        all_phases.extend(r['phases'])
        final_phases.append(r['final_phase'])
        total_wins += len(r['window_params'])

        fyrs   = r.get('fold_years', [])
        sfyrs  = r.get('saturation_fold_years', [])
        fjyrs  = r.get('fwd_jump_years', [])
        sfjyrs = r.get('saturation_fwd_jump_years', [])

        all_fold_years.extend(fyrs)
        all_sat_fold_years.extend(sfyrs)
        all_fwd_jump_years.extend(fjyrs)
        all_sat_fwd_jump_years.extend(sfjyrs)

        if sfyrs:
            n_sat_with_folds += 1
        if sfjyrs:
            n_sat_with_fwd_jumps += 1

        fxy = r.get('final_x0_year', np.nan)
        if not np.isnan(fxy):
            all_final_x0_years.append(fxy)

    # ── Saturation fold temporal clustering (backward) ─────────────────────────
    sfy = np.array(all_sat_fold_years)
    paris_sat_n,   paris_sat_p,   paris_sat_exp   = perm_test(sfy, PARIS_LO, PARIS_HI)
    covid_sat_n,   covid_sat_p,   covid_sat_exp   = perm_test(sfy, COVID_LO, COVID_HI)
    total_sat      = len(sfy)
    paris_sat_pct  = round(paris_sat_n / total_sat * 100, 1) if total_sat else 0.0
    covid_sat_pct  = round(covid_sat_n / total_sat * 100, 1) if total_sat else 0.0

    # ── Saturation forward-jump temporal clustering ────────────────────────────
    sfj = np.array(all_sat_fwd_jump_years)
    paris_satfj_n,  paris_satfj_p,  paris_satfj_exp  = perm_test(sfj, PARIS_LO, PARIS_HI)
    covid_satfj_n,  covid_satfj_p,  covid_satfj_exp  = perm_test(sfj, COVID_LO, COVID_HI)
    total_satfj     = len(sfj)
    paris_satfj_pct = round(paris_satfj_n / total_satfj * 100, 1) if total_satfj else 0.0
    covid_satfj_pct = round(covid_satfj_n / total_satfj * 100, 1) if total_satfj else 0.0

    # Also run permutation tests on ALL fold/jump years (not just saturation)
    fy_arr  = np.array(all_fold_years)
    fjy_arr = np.array(all_fwd_jump_years)
    paris_all_n,   paris_all_p,   paris_all_exp   = perm_test(fy_arr,  PARIS_LO, PARIS_HI)
    covid_all_n,   covid_all_p,   covid_all_exp   = perm_test(fy_arr,  COVID_LO, COVID_HI)
    paris_allfj_n, paris_allfj_p, paris_allfj_exp = perm_test(fjy_arr, PARIS_LO, PARIS_HI)
    covid_allfj_n, covid_allfj_p, covid_allfj_exp = perm_test(fjy_arr, COVID_LO, COVID_HI)

    # ── Inflection clusters (final window x0 in calendar years) ───────────────
    inflection_clusters = detect_inflection_clusters(all_final_x0_years)

    return {
        'label':                  label,
        'n_entities':             len(results),
        'total_windows':          total_wins,
        'median_x0':              round(np.median(all_x0), 4),
        'pct_x0_le1':             round(100 * np.mean([v <= 1.0 for v in all_x0]), 1),
        'entities_with_folds':    sum(1 for r in results.values() if r['time_folds'] > 0),
        'total_folds':            sum(r['time_folds'] for r in results.values()),
        'entities_with_fwd_jumps': sum(1 for r in results.values() if r['fwd_jumps'] > 0),
        'total_fwd_jumps':        sum(r['fwd_jumps'] for r in results.values()),
        'mean_x0_cv':             round(np.mean([r['param_stability']['x0_cv'] for r in results.values()]), 4),
        'mean_k_cv':              round(np.mean([r['param_stability']['k_cv']  for r in results.values()]), 4),
        'all_phase_dist':         pd.Series(all_phases).value_counts(normalize=True).mul(100).round(1).to_dict(),
        'final_phase_dist':       pd.Series(final_phases).value_counts(normalize=True).mul(100).round(1).to_dict(),
        # ── Saturation backward-fold stats ─────────────────────────────────────
        'total_sat_folds':        total_sat,
        'n_sat_with_folds':       n_sat_with_folds,
        'sat_fold_years':         sfy,
        'paris_sat_n':            paris_sat_n,
        'paris_sat_p':            paris_sat_p,
        'paris_sat_exp':          paris_sat_exp,
        'paris_sat_pct':          paris_sat_pct,
        'covid_sat_n':            covid_sat_n,
        'covid_sat_p':            covid_sat_p,
        'covid_sat_exp':          covid_sat_exp,
        'covid_sat_pct':          covid_sat_pct,
        # ── Saturation forward-jump stats ──────────────────────────────────────
        'total_sat_fwd_jumps':    total_satfj,
        'n_sat_with_fwd_jumps':   n_sat_with_fwd_jumps,
        'sat_fwd_jump_years':     sfj,
        'paris_satfj_n':          paris_satfj_n,
        'paris_satfj_p':          paris_satfj_p,
        'paris_satfj_exp':        paris_satfj_exp,
        'paris_satfj_pct':        paris_satfj_pct,
        'covid_satfj_n':          covid_satfj_n,
        'covid_satfj_p':          covid_satfj_p,
        'covid_satfj_exp':        covid_satfj_exp,
        'covid_satfj_pct':        covid_satfj_pct,
        # ── All fold/jump years (for reference) ────────────────────────────────
        'all_fold_years':         fy_arr,
        'all_fwd_jump_years':     fjy_arr,
        'paris_all_n':            paris_all_n,
        'paris_all_p':            paris_all_p,
        'covid_all_n':            covid_all_n,
        'covid_all_p':            covid_all_p,
        'paris_allfj_n':          paris_allfj_n,
        'paris_allfj_p':          paris_allfj_p,
        'covid_allfj_n':          covid_allfj_n,
        'covid_allfj_p':          covid_allfj_p,
        # ── Inflection cluster stats ───────────────────────────────────────────
        'final_x0_years':         all_final_x0_years,
        'inflection_clusters':    inflection_clusters,
        'n_inflection_clusters':  len(inflection_clusters),
    }


def print_comparison(summaries):
    keys = ['label', 'n_entities', 'total_windows', 'median_x0', 'pct_x0_le1',
            'entities_with_folds', 'total_folds',
            'entities_with_fwd_jumps', 'total_fwd_jumps',
            'mean_x0_cv', 'mean_k_cv']
    print(f"\n{'='*90}")
    print(f"{'Metric':<30}", end='')
    for s in summaries:
        print(f"  {s['label']:<18}", end='')
    print()
    print('-'*90)
    for k in keys:
        print(f"  {k:<28}", end='')
        for s in summaries:
            print(f"  {str(s.get(k,'')):<18}", end='')
        print()

    # ── Saturation fold section (backward) ────────────────────────────────────
    print()
    print("  Saturation-phase BACKWARD fold detection (x0 moves earlier):")
    sat_keys = [
        ('total_sat_folds',  'Total saturation folds'),
        ('n_sat_with_folds', 'Countries with sat. folds'),
        ('paris_sat_n',      'Paris 2015-16  (n)'),
        ('paris_sat_pct',    'Paris 2015-16  (%)'),
        ('paris_sat_p',      'Paris p-value (perm.)'),
        ('covid_sat_n',      'COVID 2019-21  (n)'),
        ('covid_sat_pct',    'COVID 2019-21  (%)'),
        ('covid_sat_p',      'COVID p-value (perm.)'),
    ]
    for k, label in sat_keys:
        print(f"    {label:<26}", end='')
        for s in summaries:
            val = s.get(k, '')
            print(f"  {str(val):<18}", end='')
        print()

    # ── Saturation forward-jump section ───────────────────────────────────────
    print()
    print("  Saturation-phase FORWARD jump detection (x0 moves later):")
    satfj_keys = [
        ('total_sat_fwd_jumps',  'Total sat. fwd jumps'),
        ('n_sat_with_fwd_jumps', 'Countries with sat. fwd jumps'),
        ('paris_satfj_n',        'Paris 2015-16  (n)'),
        ('paris_satfj_pct',      'Paris 2015-16  (%)'),
        ('paris_satfj_p',        'Paris p-value (perm.)'),
        ('covid_satfj_n',        'COVID 2019-21  (n)'),
        ('covid_satfj_pct',      'COVID 2019-21  (%)'),
        ('covid_satfj_p',        'COVID p-value (perm.)'),
    ]
    for k, label in satfj_keys:
        print(f"    {label:<26}", end='')
        for s in summaries:
            val = s.get(k, '')
            print(f"  {str(val):<18}", end='')
        print()

    # ── Inflection cluster section ─────────────────────────────────────────────
    print()
    print("  Inflection clusters (final-window x₀ in calendar years):")
    for s in summaries:
        clusters = s.get('inflection_clusters', [])
        print(f"    [{s['label']}] {len(clusters)} cluster(s):", end='')
        for c in clusters:
            sig = (' ***' if c.get('p', 1) < 0.001 else ' **' if c.get('p', 1) < 0.01
                   else ' *' if c.get('p', 1) < 0.05 else '')
            print(f"  {c['year']:.1f} (n={c['count']}, {c['pct']}%){sig}", end='')
        print()

    print()
    print("  Final-window phase distribution:")
    for phase in ['Acceleration', 'Inflection', 'Deceleration', 'Deactivated']:
        print(f"    {phase:<16}", end='')
        for s in summaries:
            val = s['final_phase_dist'].get(phase, 0.0)
            print(f"  {val:<18}", end='')
        print()
    print('='*90)


def plot_comparison(all_summaries, output_path):
    """
    3×3 figure comparing Approach A vs B for each window size.

    Row 1 — existing metrics:  x0 distribution | fold detection | parameter stability
    Row 2 — phase distributions per window size (3M, 6M, 12M)
    Row 3 — new analyses:
        [0] Saturation fold temporal histogram (Paris/COVID shading)
        [1] Inflection-year KDE with cluster markers
        [2] Paris & COVID clustering p-values across all conditions
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Sensitivity Analysis: Aggregated (A) vs Monthly-Fit Snapshot (B)',
                 fontsize=15, fontweight='bold', y=1.01)

    window_labels = ['3M', '6M', '12M']
    colors_A = '#3498db'
    colors_B = '#e67e22'
    width = 0.35
    x = np.arange(3)

    # ── Row 1, col 0: x0 ≤ 1 percentage ──────────────────────────────────────
    ax = axes[0, 0]
    vals_A = [all_summaries[f'A_{w}']['pct_x0_le1'] for w in window_labels]
    vals_B = [all_summaries[f'B_{w}']['pct_x0_le1'] for w in window_labels]
    ax.bar(x - width/2, vals_A, width, label='Aggregated (A)', color=colors_A, alpha=0.8)
    ax.bar(x + width/2, vals_B, width, label='Snapshot (B)',   color=colors_B, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(window_labels)
    ax.set_ylabel('% of windows'); ax.set_title('x\u2080 \u2264 1 (In-Window %)')
    ax.legend(); ax.grid(alpha=0.3)

    # ── Row 1, col 1: total folds ─────────────────────────────────────────────
    ax = axes[0, 1]
    vals_A = [all_summaries[f'A_{w}']['total_folds'] for w in window_labels]
    vals_B = [all_summaries[f'B_{w}']['total_folds'] for w in window_labels]
    ax.bar(x - width/2, vals_A, width, label='Aggregated (A)', color=colors_A, alpha=0.8)
    ax.bar(x + width/2, vals_B, width, label='Snapshot (B)',   color=colors_B, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(window_labels)
    ax.set_ylabel('Count'); ax.set_title('Total Folds Detected')
    ax.legend(); ax.grid(alpha=0.3)

    # ── Row 1, col 2: x0 CV ───────────────────────────────────────────────────
    ax = axes[0, 2]
    vals_A = [all_summaries[f'A_{w}']['mean_x0_cv'] for w in window_labels]
    vals_B = [all_summaries[f'B_{w}']['mean_x0_cv'] for w in window_labels]
    ax.bar(x - width/2, vals_A, width, label='Aggregated (A)', color=colors_A, alpha=0.8)
    ax.bar(x + width/2, vals_B, width, label='Snapshot (B)',   color=colors_B, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(window_labels)
    ax.set_ylabel('CV'); ax.set_title('x\u2080 CV (Inflection Stability)')
    ax.legend(); ax.grid(alpha=0.3)

    # ── Row 2: final-window phase distributions ───────────────────────────────
    phases       = ['Acceleration', 'Inflection', 'Deceleration', 'Deactivated']
    for col, wlabel in enumerate(window_labels):
        ax = axes[1, col]
        dist_A = [all_summaries[f'A_{wlabel}']['final_phase_dist'].get(p, 0) for p in phases]
        dist_B = [all_summaries[f'B_{wlabel}']['final_phase_dist'].get(p, 0) for p in phases]
        xi = np.arange(len(phases))
        ax.bar(xi - width/2, dist_A, width, label='Aggregated (A)', color=colors_A, alpha=0.8)
        ax.bar(xi + width/2, dist_B, width, label='Snapshot (B)',   color=colors_B, alpha=0.8)
        ax.set_xticks(xi); ax.set_xticklabels(phases, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('%'); ax.set_title(f'{wlabel}: Final-Window Phase Distribution')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Row 3, col 0: Saturation backward fold + forward jump temporal histograms
    ax = axes[2, 0]
    pooled_A_fold = np.array(all_summaries['A_6M'].get('sat_fold_years', []))
    pooled_B_fold = np.array(all_summaries['B_6M'].get('sat_fold_years', []))
    pooled_A_fwd  = np.array(all_summaries['A_6M'].get('sat_fwd_jump_years', []))
    pooled_B_fwd  = np.array(all_summaries['B_6M'].get('sat_fwd_jump_years', []))
    year_min = 1990
    year_max = 2024
    bins = np.arange(year_min, year_max + 2, 1)

    if len(pooled_A_fold) > 0:
        ax.hist(pooled_A_fold, bins=bins, alpha=0.6, color=colors_A,
                label='A-6M backward folds', edgecolor='white', linewidth=0.4)
    if len(pooled_B_fold) > 0:
        ax.hist(pooled_B_fold, bins=bins, alpha=0.6, color=colors_B,
                label='B-6M backward folds', edgecolor='white', linewidth=0.4)
    if len(pooled_A_fwd) > 0:
        ax.hist(pooled_A_fwd, bins=bins, alpha=0.4, color='green',
                label='A-6M forward jumps', edgecolor='white', linewidth=0.4,
                linestyle='--', histtype='step')
    if len(pooled_B_fwd) > 0:
        ax.hist(pooled_B_fwd, bins=bins, alpha=0.4, color='darkorange',
                label='B-6M forward jumps', edgecolor='white', linewidth=0.4,
                linestyle='--', histtype='step')

    # Shade event windows
    ax.axvspan(PARIS_LO, PARIS_HI, alpha=0.12, color='steelblue', label='Paris 2015–16')
    ax.axvspan(COVID_LO, COVID_HI,  alpha=0.12, color='firebrick',  label='COVID 2019–21')

    # Significance stars for A-6M backward folds
    s = all_summaries['A_6M']
    for (n, p, yr, clr) in [
        (s['paris_sat_n'], s['paris_sat_p'], (PARIS_LO + PARIS_HI) / 2, 'steelblue'),
        (s['covid_sat_n'], s['covid_sat_p'], (COVID_LO + COVID_HI) / 2,  'firebrick'),
    ]:
        star = ('***' if p < 0.001 else '**' if p < 0.01
                else '*' if p < 0.05 else 'n.s.')
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
        ax.text(yr, ymax * 0.92, f'↓n={n}\n{star}', ha='center', fontsize=7,
                color=clr, fontweight='bold')

    # Significance stars for A-6M forward jumps (offset slightly)
    for (n, p, yr, clr) in [
        (s['paris_satfj_n'], s['paris_satfj_p'], (PARIS_LO + PARIS_HI) / 2, 'steelblue'),
        (s['covid_satfj_n'], s['covid_satfj_p'], (COVID_LO + COVID_HI) / 2,  'firebrick'),
    ]:
        star = ('***' if p < 0.001 else '**' if p < 0.01
                else '*' if p < 0.05 else 'n.s.')
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
        ax.text(yr + 0.8, ymax * 0.72, f'↑n={n}\n{star}', ha='center', fontsize=7,
                color=clr, fontweight='bold', style='italic')

    ax.set_xlabel('Year'); ax.set_ylabel('Count')
    ax.set_title('Saturation Backward Folds & Forward Jumps\n(6M baseline, A-6M stars)',
                 fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_xlim(year_min, year_max)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Row 3, col 1: Inflection-year KDE with cluster markers ────────────────
    ax = axes[2, 1]
    # Use A-6M and B-6M final x0 years
    for key, color, lbl in [('A_6M', colors_A, 'A-6M'), ('B_6M', colors_B, 'B-6M')]:
        fyears = all_summaries[key].get('final_x0_years', [])
        fyears = [y for y in fyears if not np.isnan(y)]
        if len(fyears) < 3:
            continue
        iy = np.array(fyears)
        bw = max(2.0 / (iy.std() + 1e-9), 0.05)
        kde  = gaussian_kde(iy, bw_method=bw)
        grid = np.linspace(year_min, year_max, 500)
        dens = kde(grid)
        ax.plot(grid, dens, color=color, lw=2, label=f'{lbl} KDE')
        ax.fill_between(grid, dens, alpha=0.15, color=color)

        # Mark cluster peaks
        clusters = all_summaries[key].get('inflection_clusters', [])
        for c in clusters:
            yr = c['year']
            d  = kde(np.array([yr]))[0]
            ax.annotate(f"{yr:.0f}\n(n={c['count']})",
                        xy=(yr, d), xytext=(yr, d * 1.12),
                        ha='center', fontsize=7.5, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.axvspan(PARIS_LO, PARIS_HI, alpha=0.10, color='steelblue', label='Paris 2015–16')
    ax.axvspan(COVID_LO, COVID_HI,  alpha=0.10, color='firebrick',  label='COVID 2019–21')
    ax.set_xlabel('Inflection Year (x\u2080 → calendar year)')
    ax.set_ylabel('Density')
    ax.set_title('Inflection Cluster Distribution (6M baseline)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_xlim(year_min, year_max)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Row 3, col 2: Paris & COVID p-values across all conditions ────────────
    ax = axes[2, 2]
    cond_labels = [f'A-{w}' for w in window_labels] + [f'B-{w}' for w in window_labels]
    cond_keys   = [f'A_{w}' for w in window_labels] + [f'B_{w}' for w in window_labels]
    paris_ps = [all_summaries[k].get('paris_sat_p', np.nan) for k in cond_keys]
    covid_ps  = [all_summaries[k].get('covid_sat_p',  np.nan) for k in cond_keys]

    xi = np.arange(len(cond_labels))
    ax.bar(xi - width/2, [-np.log10(max(p, 1e-4)) for p in paris_ps], width,
           label='Paris 2015–16', color='steelblue', alpha=0.8)
    ax.bar(xi + width/2, [-np.log10(max(p, 1e-4)) for p in covid_ps],  width,
           label='COVID 2019–21', color='firebrick',  alpha=0.8)
    ax.axhline(-np.log10(0.05), color='black', ls='--', lw=1, label='p=0.05')
    ax.axhline(-np.log10(0.01), color='gray',  ls=':',  lw=1, label='p=0.01')
    ax.set_xticks(xi); ax.set_xticklabels(cond_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('−log₁₀(p)')
    ax.set_title('Saturation Fold Cluster Significance\n(Paris & COVID)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('='*70)
    print('SENSITIVITY ANALYSIS COMPARISON: Aggregated vs Snapshot')
    print('='*70)

    # Load and prepare monthly data
    print('\nLoading data...')
    df_raw = pd.read_csv('pivoted_data_states_monthly.csv')
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw[df_raw['date'] <= '2023-07-31'].copy()
    print(f"  Raw rows: {len(df_raw)}")

    # Aggregate to proper monthly (sums all publications per calendar month)
    df_raw.set_index('date', inplace=True)
    df_monthly = df_raw.resample('MS').sum().reset_index()
    print(f"  Monthly rows: {len(df_monthly)}  "
          f"({df_monthly['date'].iloc[0].date()} to {df_monthly['date'].iloc[-1].date()})")

    # Group entities
    print('\nGrouping entities...')
    df_monthly = group_entities(df_monthly)

    # Standardise to TOTAL_MONTHS = PAD_MONTHS + DATA_MONTHS (100 + 199 = 299)
    print('\nStandardising series length...')
    df_monthly = standardise_monthly_series(df_monthly)

    # ── Run all conditions ────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('APPROACH A — Aggregated fitting')
    print('='*70)
    res_A = {}
    for wm in [3, 6, 12]:
        res_A[f'A_{wm}M'] = run_aggregated(df_monthly, wm)

    print('\n' + '='*70)
    print('APPROACH B — Monthly fit, snapshot at intervals')
    print('='*70)
    res_B = {}
    for wm in [3, 6, 12]:
        res_B[f'B_{wm}M'] = run_snapshot(df_monthly, wm)

    # ── Summarise ─────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)

    all_summaries = {}
    for wm in ['3M', '6M', '12M']:
        all_summaries[f'A_{wm}'] = summarise(res_A[f'A_{wm}'], f'A-{wm}')
        all_summaries[f'B_{wm}'] = summarise(res_B[f'B_{wm}'], f'B-{wm}')

    for wm in ['3M', '6M', '12M']:
        print(f'\n--- {wm} ---')
        print_comparison([all_summaries[f'A_{wm}'], all_summaries[f'B_{wm}']])

    # Cross-window comparison within each approach
    print('\n--- Cross-window: Approach A ---')
    print_comparison([all_summaries[f'A_{w}'] for w in ['3M','6M','12M']])
    print('\n--- Cross-window: Approach B ---')
    print_comparison([all_summaries[f'B_{w}'] for w in ['3M','6M','12M']])

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    # Fields that contain arrays/lists are not suitable for flat CSV rows
    _SKIP_CSV = {'all_phase_dist', 'final_phase_dist',
                 'sat_fold_years', 'all_fold_years', 'final_x0_years',
                 'inflection_clusters'}
    rows = []
    for key, s in all_summaries.items():
        row = {k: v for k, v in s.items() if k not in _SKIP_CSV}
        for phase in ['Acceleration', 'Inflection', 'Deceleration', 'Deactivated']:
            row[f'final_{phase}'] = s['final_phase_dist'].get(phase, 0.0)
        # Flatten inflection clusters (top 3)
        for ci, c in enumerate(s.get('inflection_clusters', [])[:3]):
            row[f'cluster{ci+1}_year']  = c['year']
            row[f'cluster{ci+1}_count'] = c['count']
            row[f'cluster{ci+1}_pct']   = c['pct']
        rows.append(row)
    pd.DataFrame(rows).to_csv('sensitivity_comparison_AB.csv', index=False)
    print('\nSaved: sensitivity_comparison_AB.csv')

    # ── Print saturation fold + cluster summary ────────────────────────────────
    print('\n' + '='*70)
    print('SATURATION FOLD & INFLECTION CLUSTER SUMMARY')
    print('='*70)
    for wm in ['3M', '6M', '12M']:
        print(f'\n--- {wm} ---')
        for appr in ['A', 'B']:
            s = all_summaries[f'{appr}_{wm}']
            print(f"  [{appr}]  sat_folds={s['total_sat_folds']}  "
                  f"paris: n={s['paris_sat_n']} p={s['paris_sat_p']}  "
                  f"covid: n={s['covid_sat_n']} p={s['covid_sat_p']}  "
                  f"infl_clusters={s['n_inflection_clusters']}")
            for c in s['inflection_clusters']:
                print(f"       cluster @ {c['year']:.1f}: n={c['count']} ({c['pct']}%)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print('\nGenerating comparison plot...')
    plot_comparison(all_summaries, 'sensitivity_comparison_AB.png')

    print('\nDone.')
