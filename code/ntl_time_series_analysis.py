# -*- coding: utf-8 -*-

"""
ntl_time_series_analysis.py
===========================

Compute two nighttime lights (NTL) change metrics for each Arctic settlement:

    1. collapse_loss_pct   - how far NTL fell from the early-1990s baseline
                             to its lowest observed value (the trough).
    2. recent_vs_early_pct - how NTL in the early 2010s compares with the
                             early-1990s baseline.

Both metrics are expressed as a percentage of the early baseline, so
settlements of very different sizes and brightness can be compared directly.


INPUT
-----
    outputs/ntl_analysis_sett/ntl_timeseries_sett_v2.csv

    One row per settlement per year (long format). Required fields:
        sett     - settlement identifier
        year     - observation year (integer)
        ntl_sum  - summed nighttime lights radiance for the settlement in
                   that year

    Rows with a missing or non-numeric value in any of these fields are
    dropped. Byte-order marks (BOMs) and whitespace are stripped from
    column names.


OUTPUT
------
    outputs/ntl_analysis_sett/ntl_collapse_recovery_metrics_v2.csv

    One row per settlement, sorted by recent_vs_early_pct (highest first,
    NaN last). Written as UTF-8 with BOM so it opens cleanly in Excel.

    Columns:
        sett                     - settlement identifier
        early_median_1992_1995   - median ntl_sum over 1992-1995 (baseline)
        trough_year              - year of the minimum ntl_sum across the
                                   whole series (first such year if tied)
        trough_ntl               - minimum ntl_sum across the whole series
        recent_median_2010_2013  - median ntl_sum over 2010-2013
        collapse_loss_abs        - early_median - trough_ntl
        collapse_loss_pct        - 100 * collapse_loss_abs / early_median
        recent_vs_early_abs      - recent_median - early_median
        recent_vs_early_pct      - 100 * recent_vs_early_abs / early_median


METHOD AND DEFINITIONS
----------------------
    Early baseline (EARLY_YEARS = 1992-1995)
        The median rather than a single year is used so that one noisy or
        anomalous year cannot drive the baseline. If a settlement has no
        observations in this window, the baseline is NaN.

    Recent level (RECENT_YEARS = 2010-2013)
        Median over the final four years, for the same robustness reason.

    Trough
        The single lowest ntl_sum value in the settlement's full series. It
        is not limited to years after the baseline window, so it can fall
        inside 1992-1995.

    collapse_loss_pct
        Positive values mean NTL fell below the baseline (a larger value
        means a deeper collapse). Because the trough is the series minimum,
        it can never be above the early median, so this value is >= 0
        whenever it is defined. If the loss is within EPS of zero (the
        baseline itself is the trough), both collapse_loss_abs and
        collapse_loss_pct are set to exactly 0.

    recent_vs_early_pct
        Positive values mean the settlement is brighter in 2010-2013 than in
        1992-1995, negative values mean it is dimmer. As ntl_sum is
        non-negative, values below -100 are not possible, but there is no
        upper limit.

    Undefined values
        Both percentage metrics are NaN when the early baseline is missing
        or effectively zero (|early_median| <= EPS), because a percentage of
        a zero baseline is undefined. recent_vs_early_pct is also NaN when
        the settlement has no observations in 2010-2013.


CONSOLE OUTPUT
--------------
    - Number of settlements and rows written.
    - Count of missing (NaN) values for the key columns, as a quick check
      on baseline and recent-window coverage.
    - A preview of the first 20 rows.


DOWNSTREAM USE
--------------
    The output file is read by code/models/lr_pred_and_assess.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =====================================================
# FILES
# =====================================================

IN_FILE = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\ntl_timeseries_sett_v2.csv"
)

OUT_FILE = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\ntl_collapse_recovery_metrics_v2.csv"
)


# =====================================================
# SETTINGS
# =====================================================

SETT_COL = "sett"
YEAR_COL = "year"
NTL_COL = "ntl_sum"

EARLY_YEARS = range(1992, 1996)   # 1992–1995
RECENT_YEARS = range(2010, 2014)  # 2010–2013

EPS = 1e-9


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def safe_float(x):
    if pd.isna(x):
        return np.nan
    return float(x)


def pct_change(numerator, denominator):
    """
    Return 100 * numerator / denominator when denominator is nonzero.
    Otherwise return NaN.
    """
    if pd.notna(denominator) and abs(denominator) > EPS:
        return 100.0 * numerator / denominator
    return np.nan


# =====================================================
# READ AND CLEAN
# =====================================================

print("[INFO] Reading cleaned NTL time series...")

df = pd.read_csv(IN_FILE)

df.columns = (
    df.columns
    .str.replace("ï»¿", "", regex=False)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

required = [SETT_COL, YEAR_COL, NTL_COL]
missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing required fields: {missing}")

df[SETT_COL] = df[SETT_COL].astype(str).str.strip()
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df[NTL_COL] = pd.to_numeric(df[NTL_COL], errors="coerce")

df = df.dropna(subset=[SETT_COL, YEAR_COL, NTL_COL]).copy()
df[YEAR_COL] = df[YEAR_COL].astype(int)

df = df.sort_values([SETT_COL, YEAR_COL]).reset_index(drop=True)


# =====================================================
# METRIC FUNCTION
# =====================================================

def calculate_metrics(g):
    sett = g[SETT_COL].iloc[0]

    early = g[g[YEAR_COL].isin(EARLY_YEARS)]
    recent = g[g[YEAR_COL].isin(RECENT_YEARS)]

    early_median = safe_float(early[NTL_COL].median())
    recent_median = safe_float(recent[NTL_COL].median())

    min_idx = g[NTL_COL].idxmin()
    trough_year = int(g.loc[min_idx, YEAR_COL])
    trough_ntl = safe_float(g.loc[min_idx, NTL_COL])

    # -----------------------------------------------------
    # Collapse loss
    # -----------------------------------------------------

    collapse_loss_abs = early_median - trough_ntl
    collapse_loss_pct = pct_change(collapse_loss_abs, early_median)

    # If early and trough are effectively the same, there is no observed loss.
    if pd.notna(collapse_loss_abs) and abs(collapse_loss_abs) <= EPS:
        collapse_loss_abs = 0.0
        collapse_loss_pct = 0.0

    # -----------------------------------------------------
    # Recent versus early
    # -----------------------------------------------------

    recent_vs_early_abs = recent_median - early_median
    recent_vs_early_pct = pct_change(recent_vs_early_abs, early_median)

    return pd.Series({
        "sett": sett,

        "early_median_1992_1995": early_median,
        "trough_year": trough_year,
        "trough_ntl": trough_ntl,
        "recent_median_2010_2013": recent_median,

        "collapse_loss_abs": collapse_loss_abs,
        "collapse_loss_pct": collapse_loss_pct,

        "recent_vs_early_abs": recent_vs_early_abs,
        "recent_vs_early_pct": recent_vs_early_pct,
    })


# =====================================================
# CALCULATE
# =====================================================

print("[INFO] Calculating median-based NTL collapse loss and recent-vs-early change...")

metrics = (
    df.groupby(SETT_COL, group_keys=False)
    .apply(calculate_metrics)
    .reset_index(drop=True)
)

metrics = metrics.sort_values(
    "recent_vs_early_pct",
    ascending=False,
    na_position="last"
).reset_index(drop=True)


# =====================================================
# SAVE
# =====================================================

metrics.to_csv(
    OUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n[OK] Saved: {OUT_FILE}")
print(f"[INFO] Settlements: {metrics['sett'].nunique():,}")
print(f"[INFO] Rows: {len(metrics):,}")

print("\n[MISSINGNESS CHECK]")
check_cols = [
    "early_median_1992_1995",
    "trough_ntl",
    "recent_median_2010_2013",
    "collapse_loss_pct",
    "recent_vs_early_pct",
]
print(metrics[check_cols].isna().sum().to_string())

print("\n[PREVIEW]")
print(metrics.head(20).to_string(index=False))
