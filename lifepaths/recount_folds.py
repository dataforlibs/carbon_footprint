"""
Recount backward time folds from the two lifepath point files.

Fold rule: for one entity, order its points by snapshot date (yr). A fold is a
step between consecutive points where the projected inflection date (x) moves
BACK by more than THRESHOLD_DAYS (default 183).

Usage (run from the folder that holds the JSON files):
    python3 recount_folds.py
    python3 recount_folds.py --days 183 --dir /path/to/carbon_footprint/lifepaths
"""
import argparse, json, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=".", help="folder with the JSON files")
ap.add_argument("--days", type=float, default=183, help="backward threshold in days")
a = ap.parse_args()

COUNTRIES = "all_years_shipped_to_final_saturation_final_circles3_Gore1_states.json"
AUTHORS   = "all_years_shipped_to_final_saturation_final_circles3_Gore1.json"
CUSTOMERS = "customers_more1_Gore_chem.json"      # list of authors kept in the paper


def load(name):
    with open(os.path.join(a.dir, name)) as f:
        return json.load(f)


def recount(points_by_entity, keep=None):
    entities = [k for k in points_by_entity if keep is None or k in keep]
    total, with_fold, mags = 0, 0, []
    for k in entities:
        pts = sorted((pd.Timestamp(r["yr"]), pd.Timestamp(r["x"])) for r in points_by_entity[k])
        n = 0
        for (_, x0), (_, x1) in zip(pts, pts[1:]):
            back = (x0 - x1).days            # positive = x moved earlier
            if back > a.days:
                n += 1
                mags.append(back / 30.44)    # months
        total += n
        with_fold += n > 0
    return len(entities), total, with_fold, np.array(mags)


def report(label, N, total, with_fold, mags):
    print(f"{label}")
    print(f"  entities analysed        : {N}")
    print(f"  backward folds           : {total}")
    print(f"  entities with >=1 fold   : {with_fold}  ({100 * with_fold / N:.1f}%)")
    if len(mags):
        print(f"  fold magnitude (months)  : mean {mags.mean():.1f}, median {np.median(mags):.1f}")
    print()


print(f"Rule: x moves back by more than {a.days:g} days between consecutive snapshots\n")

countries = load(COUNTRIES)
report("COUNTRIES", *recount(countries))

authors = load(AUTHORS)
keep = set(load(CUSTOMERS))                       # the 7,960 authors in the paper
keep &= set(authors)
report("AUTHORS (in both the point file and the customers file)", *recount(authors, keep))
