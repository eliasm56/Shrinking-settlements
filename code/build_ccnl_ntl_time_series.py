# -*- coding: utf-8 -*-

"""
Build and save cleaned NTL time series.

Input:
    Folder of annual/per-year CCNL NTL CSVs

Output:
    ntl_timeseries_clean.csv

Expected fields:
    sett
    year
    ntl_sum
"""

from pathlib import Path
import pandas as pd


# =====================================================
# SETTINGS
# =====================================================

NTL_DIR = Path(r"D:\PhD_main\chapter_3\data\time_series_v3\ntl_ccnl_settlements")

OUT_DIR = Path(r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEOG_COL = "sett"
YEAR_COL = "year"
NTL_COL = "ntl_sum"

START_YEAR = 1992
END_YEAR = 2013


# =====================================================
# HELPERS
# =====================================================

def read_csv_folder(folder):
    csvs = sorted(folder.glob("*.csv"))

    if not csvs:
        raise RuntimeError(f"No CSV files found in: {folder}")

    df = pd.concat(
        [pd.read_csv(fp) for fp in csvs],
        ignore_index=True
    )

    return df


def clean_ntl_timeseries(df):
    df = df.copy()

    # Fix possible UTF-8 BOM / corrupted header issue
    df.columns = (
        df.columns
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )

    required = [GEOG_COL, YEAR_COL, NTL_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required fields: {missing}")

    df[GEOG_COL] = df[GEOG_COL].astype(str).str.strip()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[NTL_COL] = pd.to_numeric(df[NTL_COL], errors="coerce").fillna(0)

    df = df.dropna(subset=[GEOG_COL, YEAR_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    df = df[
        (df[YEAR_COL] >= START_YEAR) &
        (df[YEAR_COL] <= END_YEAR)
        ].copy()

    # -----------------------------------------------------
    # Remove settlements/regions with no detectable NTL
    # -----------------------------------------------------

    peak_ntl = (
        df.groupby(GEOG_COL)[NTL_COL]
        .max()
    )

    valid_geogs = peak_ntl[peak_ntl > 0].index

    n_removed = df[GEOG_COL].nunique() - len(valid_geogs)

    print(
        f"[INFO] Removing {n_removed:,} "
        f"{GEOG_COL}s with peak NTL = 0"
    )

    df = df[df[GEOG_COL].isin(valid_geogs)].copy()

    df = df.sort_values([GEOG_COL, YEAR_COL]).reset_index(drop=True)
    return df


# =====================================================
# MAIN
# =====================================================

print("[INFO] Reading NTL CSV files...")
ntl_raw = read_csv_folder(NTL_DIR)

print("[INFO] Cleaning NTL time series...")
ntl_clean = clean_ntl_timeseries(ntl_raw)

out_csv = OUT_DIR / "ntl_timeseries_sett_v2.csv"
ntl_clean.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"[OK] Wrote: {out_csv}")
print(f"[INFO] Rows: {len(ntl_clean):,}")
print(f"[INFO] Geographies: {ntl_clean[GEOG_COL].nunique():,}")
print(f"[INFO] Years: {ntl_clean[YEAR_COL].min()}–{ntl_clean[YEAR_COL].max()}")

print("\n[PREVIEW]")
print(ntl_clean.head(20))