#!/bin/sh
# Run from inside this folder.  JOBS = number of CPU cores to use (default 4).
JOBS=${1:-4}
python run_grid.py --csv pivoted_data_states_monthly.csv --out base_fits --jobs $JOBS &&
python build_dashboard_data.py --fits base_fits --out dashboard_data &&
python check_against_paper.py --data dashboard_data &&
python check_lifepaths.py dashboard_data/sens_grid/pad100_docs0_months1/aggregated_fit/6M/country_results.json --countries France USA India Brazil --out lifepath_check_primary.png &&
python check_lifepaths.py dashboard_data/sens_grid/pad0_docs0_months1/aggregated_fit/6M/country_results.json --countries France USA India Brazil --out lifepath_check_baseline.png
