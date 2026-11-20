# Country sensitivity dashboard: data pipeline

Everything needed to produce the dashboard data is in this folder. Needs Python with numpy, pandas, scipy, matplotlib.

    sh run_all.sh 4        # 4 = number of CPU cores

or step by step:

1. `python run_grid.py --jobs 4`  
   Fits the 12 base conditions (padding 0/100 x approach A/B x window 3/6/12 months) with the fitting and phase code of
   `sensitivity_analysis_comparison_2_1_2_1_with_forward.py` (imported, not copied), and saves every window fit to `base_fits/`.
   This is the only slow step (the 3M and B conditions take longest; about 3-4 minutes per condition per core was measured on
   the simplest one). It is resumable: finished conditions are skipped. Test one condition first: `python run_grid.py --only 0_A_12`.
2. `python build_dashboard_data.py`  
   Derives all 288 scenarios (12 conditions x min documents 0-5 x min months 1/6/12/24) from the saved fits, with no refitting,
   and writes `dashboard_data/manifest.json`, `scenario_summaries.json` and `sens_grid/.../country_results.json`
   (same format as lifepaths_output2, so the existing lifepath plot code reads it unchanged).
   By default country files are written for the 12 base conditions plus the document and month sweeps from the two A-6M
   conditions (`--country-files all` writes all 288).
3. `python check_against_paper.py`  
   Compares every condition with Tables S1 and S2 and Section 4.2.5, and prints OK or DIFF for each number.
4. `python check_lifepaths.py <country_results.json> --countries France USA India Brazil`  
   Draws lifepaths with the rules of the fixed page (x = x0_year, y = ln k with the axis reversed, windows with x0 >= 1.19, k > 3000 or A/total > 50 skipped, phase colours, backward folds solid red, forward folds dashed orange).

## Rules, so the numbers can be reproduced
- A fit never depends on where the analysis starts, so min documents and min months only choose which windows are kept.
  Min documents: keep a window if cumulative publications >= the value (0 = keep all, as in the May runs).
  Min months: keep a window if at least that many months have passed since the country's first publication (1 = no rule).
- Folds are re-detected on the kept windows: x0 <= 1 in both windows and a step larger than 1/(n_data-1).
- The paper's tables count only windows from each country's first publication and average the CV over every country with a fit.
  The summaries store both the all-windows numbers and this "paper_definition".
- Permutation p-values use a fixed random seed per test, so they can differ from the May runs in the third decimal.
