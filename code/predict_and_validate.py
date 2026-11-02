# -*- coding: utf-8 -*-

"""
Apply the final logistic regression to ALL settlements and assess map accuracy.

Purpose
-------
This script applies the finalized logistic-regression workflow for predicting
severe settlement population decline from artificial-light-at-night (ALAN)
collapse/recovery metrics.

The model is NOT fitted here. It is fitted once, on the full training dataset,
by code/models/logistic_regression_single_model.py, which saves its weights to:
    logistic_two_variable_cv/two_variable_full_model_weights.json

Run that script first whenever the training data, predictors or
severe-decline threshold change.

Workflow
--------
1. Load the saved model weights (intercept, coefficients and the
   StandardScaler mean/scale for each predictor).

2. Check that the saved model matches this script's settings:
       - the predictors and their order must equal PREDICTORS
       - the severe-decline threshold used to label the training data must
         equal SEVERE_DECLINE_THRESHOLD
   The script stops with an error if either differs.

3. (no refitting) Predicted probability for each row is
       z = (x - scaler_mean) / scaler_scale
       p = 1 / (1 + exp(-(intercept + sum(coefficient * z))))

4. Apply the saved model to the full NTL metrics dataset:
       ntl_collapse_recovery_metrics.csv

   This produces the map-ready prediction table:
       ntl_all_settlements_logistic_predictions.csv

5. Read the independent master validation dataset:
       master_matched_validation_with_all_metrics.csv

6. Recalculate validation true labels from:
       percent_decrease <= SEVERE_DECLINE_THRESHOLD

   This ensures the same severe-decline threshold is applied consistently to
   both training and validation labels.

7. Apply the same saved model to the validation rows.

8. Assess accuracy using the fixed probability threshold only:
       PROB_THRESHOLD

Outputs
-------
Full settlement prediction outputs:
    ntl_all_settlements_logistic_predictions.csv
    ntl_all_settlements_logistic_scaler_params.csv
        (scaler mean/std and coefficient per predictor, copied from the
        weights file for reference)

The full statsmodels summary of the model is written by the training script
(two_variable_full_model_summary.txt), not here.

Validation outputs:
    logistic_master_validation_predictions.csv
    logistic_master_validation_metrics.csv
    logistic_master_validation_confusion_matrix.csv
    logistic_master_validation_classification_report.csv
    logistic_master_validation_run_summary.csv

Important methodological note
-----------------------------
This script intentionally does NOT test a range of probability thresholds on
the validation dataset. Threshold testing belongs in the model-selection script,
where thresholds are selected using out-of-fold training predictions only.

The independent validation dataset should be used only once to evaluate the
already-selected model and already-selected threshold.

Label convention
----------------
1 = Severe decline
0 = Not severe decline

Zeros in predictor columns are preserved as valid numeric values. Only blank,
non-numeric, NaN, or infinite values are treated as missing.
"""

import os

os.environ["PROJ_LIB"] = (
    r"C:\Users\etm17002\Miniconda3\envs\osm\Library\share\proj"
)
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =====================================================
# FILE PATHS
# =====================================================

# Written by code/models/logistic_regression_single_model.py
MODEL_WEIGHTS_FILE = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\logistic_two_variable_cv\two_variable_full_model_weights.json"
)

PREDICT_FILE = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\ntl_collapse_recovery_metrics_v2.csv"
)

MASTER_MATCHED_FILE = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\accuracy_assessment\master_matched_validation_with_all_metrics.csv"
)

SETT_POINTS_FILE = Path(
    r"D:\PhD_main\chapter_3\data\rus_sett_DCW_ACPR.shp"
)

SETT_BUFFERS_FILE = Path(
    r"D:\PhD_main\chapter_3\outputs\gis\named_sett_500m_dissolved_buffers.shp"
)

OUT_ALL_PRED_DIR = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\logistic_prediction_all_settlements_v3"
)
OUT_ALL_PRED_DIR.mkdir(parents=True, exist_ok=True)

OUT_ASSESS_DIR = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\accuracy_assessment\logistic_regression_v3"
)
OUT_ASSESS_DIR.mkdir(parents=True, exist_ok=True)

# Full prediction outputs
OUT_ALL_PREDICTIONS = OUT_ALL_PRED_DIR / "ntl_all_settlements_logistic_predictions.csv"
OUT_SCALER_PARAMS = OUT_ALL_PRED_DIR / "ntl_all_settlements_logistic_scaler_params.csv"

OUT_POINTS_GPKG = OUT_ALL_PRED_DIR / "ntl_all_settlements_logistic_predictions_points.gpkg"
OUT_BUFFERS_GPKG = OUT_ALL_PRED_DIR / "ntl_all_settlements_logistic_predictions_buffers.gpkg"
OUT_SPATIAL_JOIN_SUMMARY = OUT_ALL_PRED_DIR / "ntl_all_settlements_logistic_predictions_spatial_join_summary.csv"

# Validation accuracy outputs
OUT_VALIDATION_PREDICTIONS = OUT_ASSESS_DIR / "logistic_master_validation_predictions.csv"
OUT_METRICS = OUT_ASSESS_DIR / "logistic_master_validation_metrics.csv"
OUT_CONFUSION = OUT_ASSESS_DIR / "logistic_master_validation_confusion_matrix.csv"
OUT_CLASS_REPORT = OUT_ASSESS_DIR / "logistic_master_validation_classification_report.csv"
OUT_RUN_SUMMARY = OUT_ASSESS_DIR / "logistic_master_validation_run_summary.csv"


# =====================================================
# SETTINGS
# =====================================================

SETT_COL = "sett"

# Training population-change column.
TRAIN_POP_CHANGE_COL = "pop_decrease"

# Validation population-change column.
# This should be in percentage units, e.g. -42.5 means -42.5%.
VALIDATION_PERCENT_CHANGE_COL = "percent_decrease"

TRAIN_BINARY_COL = "severe_decline_binary"

TRUE_BINARY_COL = "true_severe_decline_binary"
TRUE_CLASS_COL = "true_severe_decline_class"

PROB_COL = "prob_severe_decline"
PRED_BINARY_COL = "pred_binary"
PRED_CLASS_COL = "predicted_class"

# Same severe-decline threshold used for training and validation labels.
SEVERE_DECLINE_THRESHOLD = -50.0

# Model probability threshold
PROB_THRESHOLD = 0.50

# Final selected logistic predictors.
# These must match the model-selection output.
PREDICTORS = [
    "collapse_loss_pct",
    "recent_vs_early_pct",
]


# =====================================================
# HELPERS
# =====================================================

def clean_columns(df):
    """
    Clean common CSV encoding artifacts and whitespace in column names.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace("ï»¿", "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def clean_numeric_keep_zero(series):
    """
    Convert to numeric while keeping 0 as a valid value.

    Handles:
        - unicode minus signs
        - percent signs
        - comma separators
        - common null strings
        - infinite values
    """
    s = series.copy()

    if s.dtype == object:
        s = (
            s.astype(str)
            .str.strip()
            .str.replace("\u2212", "-", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .replace(
                {
                    "": np.nan,
                    "nan": np.nan,
                    "NaN": np.nan,
                    "None": np.nan,
                    "none": np.nan,
                    "NULL": np.nan,
                    "null": np.nan,
                    "<NA>": np.nan,
                }
            )
        )

    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    return s


def check_required_columns(df, required_cols, file_label):
    """
    Raise an error if required columns are missing.
    """
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"\nMissing required columns in {file_label}: {missing}\n\n"
            f"Available columns:\n{df.columns.tolist()}"
        )


def resolve_predictor_columns(df, predictors):
    """
    Find predictor columns in a dataframe.

    In the master matched file, columns may be unsuffixed or suffixed from a
    merge. Unsuffixed metric columns are preferred.
    """
    resolved = {}

    for predictor in predictors:
        candidates = [
            predictor,
            f"{predictor}_metrics",
            f"{predictor}_pred",
            f"{predictor}_true",
        ]

        found = [c for c in candidates if c in df.columns]

        if not found:
            raise RuntimeError(
                f"Could not find predictor `{predictor}`.\n"
                f"Tried: {candidates}\n"
                f"Available columns:\n{df.columns.tolist()}"
            )

        resolved[predictor] = found[0]

    return resolved


def add_validation_true_labels(validation_df):
    """
    Recalculate validation true labels from the validation population-change field.

    This intentionally ignores stale pre-existing true-label columns and rebuilds
    labels using SEVERE_DECLINE_THRESHOLD.
    """
    df = validation_df.copy()

    if VALIDATION_PERCENT_CHANGE_COL not in df.columns:
        raise RuntimeError(
            f"Master validation file is missing `{VALIDATION_PERCENT_CHANGE_COL}`.\n"
            "The validation file must contain this column so the same "
            "severe-decline threshold can be applied to validation labels."
        )

    df[VALIDATION_PERCENT_CHANGE_COL] = clean_numeric_keep_zero(
        df[VALIDATION_PERCENT_CHANGE_COL]
    )

    df[TRUE_BINARY_COL] = np.where(
        df[VALIDATION_PERCENT_CHANGE_COL] <= SEVERE_DECLINE_THRESHOLD,
        1,
        np.where(df[VALIDATION_PERCENT_CHANGE_COL].notna(), 0, np.nan)
    )

    df[TRUE_CLASS_COL] = np.where(
        df[TRUE_BINARY_COL] == 1,
        "Severe decline",
        np.where(df[TRUE_BINARY_COL] == 0, "Not severe decline", pd.NA)
    )

    df["validation_label_threshold_used"] = SEVERE_DECLINE_THRESHOLD
    df["validation_label_source_column"] = VALIDATION_PERCENT_CHANGE_COL

    return df


# =====================================================
# MODEL WEIGHTS
# =====================================================

def load_model_weights(path):
    """
    Load the saved logistic model and check it matches this script's settings.
    """
    if not path.exists():
        raise RuntimeError(
            f"Model weights file not found:\n  {path}\n"
            "Run code/models/logistic_regression_single_model.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    if list(model["predictors"]) != list(PREDICTORS):
        raise RuntimeError(
            "Saved model predictors do not match PREDICTORS.\n"
            f"  Saved:  {model['predictors']}\n"
            f"  Script: {PREDICTORS}\n"
            "Refit the model or update PREDICTORS."
        )

    if not np.isclose(model["severe_decline_threshold"], SEVERE_DECLINE_THRESHOLD):
        raise RuntimeError(
            "Saved model was trained with a different severe-decline threshold.\n"
            f"  Saved:  {model['severe_decline_threshold']}\n"
            f"  Script: {SEVERE_DECLINE_THRESHOLD}\n"
            "Refit the model or update SEVERE_DECLINE_THRESHOLD so training "
            "and validation labels use the same threshold."
        )

    if not model.get("converged", True):
        print("[WARNING] The saved model did not converge when it was fitted.")

    return model


def predict_proba(X_raw, model):
    """
    Predicted probability of severe decline from raw (unscaled) predictors.
    """
    mean = np.array([model["scaler_mean"][p] for p in PREDICTORS])
    scale = np.array([model["scaler_scale"][p] for p in PREDICTORS])
    coefs = np.array([model["coefficients"][p] for p in PREDICTORS])

    z = (X_raw[PREDICTORS].to_numpy(dtype=float) - mean) / scale
    linear = model["intercept"] + z @ coefs

    return pd.Series(1.0 / (1.0 + np.exp(-linear)), index=X_raw.index)


# =====================================================
# PREDICTION
# =====================================================

def apply_logistic_model(df, model, source_label):
    """
    Apply the saved logistic model to any dataframe containing the predictors.

    Rows with missing predictors are retained in the output but receive missing
    prediction fields.
    """
    out = df.copy()

    if SETT_COL in out.columns:
        out[SETT_COL] = out[SETT_COL].astype(str).str.strip()

    predictor_map = resolve_predictor_columns(out, PREDICTORS)

    X_raw = pd.DataFrame(index=out.index)

    for predictor in PREDICTORS:
        source_col = predictor_map[predictor]
        X_raw[predictor] = clean_numeric_keep_zero(out[source_col])

    valid_predictor_mask = X_raw.notna().all(axis=1)

    out[PROB_COL] = np.nan
    out[PRED_BINARY_COL] = np.nan
    out[PRED_CLASS_COL] = pd.NA
    out["prediction_status"] = "missing predictor value"
    out["prediction_source"] = source_label

    if valid_predictor_mask.sum() > 0:
        pred_prob = predict_proba(X_raw.loc[valid_predictor_mask], model)
        pred_binary = (pred_prob >= PROB_THRESHOLD).astype(int)

        out.loc[valid_predictor_mask, PROB_COL] = pred_prob
        out.loc[valid_predictor_mask, PRED_BINARY_COL] = pred_binary
        out.loc[valid_predictor_mask, PRED_CLASS_COL] = np.where(
            pred_binary == 1,
            "Severe decline",
            "Not severe decline"
        )
        out.loc[valid_predictor_mask, "prediction_status"] = "predicted"

    # Add exact model input values for transparency.
    for predictor in PREDICTORS:
        out[f"logit_input_{predictor}"] = X_raw[predictor]

    return out, predictor_map, valid_predictor_mask


def order_prediction_columns(df):
    """
    Put the most important fields first while preserving all other columns.
    """
    first_cols = [
        SETT_COL,
        "sett_true",
        VALIDATION_PERCENT_CHANGE_COL,
        TRUE_BINARY_COL,
        TRUE_CLASS_COL,
        PROB_COL,
        PRED_BINARY_COL,
        PRED_CLASS_COL,
        "prediction_status",
        "prediction_source",
        "validation_label_threshold_used",
        "validation_label_source_column",
    ] + PREDICTORS + [f"logit_input_{p}" for p in PREDICTORS]

    first_cols = [c for c in first_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in first_cols]

    return df[first_cols + remaining_cols]


# =====================================================
# ACCURACY
# =====================================================

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate fixed-threshold validation metrics.
    """
    metrics = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_severe": precision_score(y_true, y_pred, zero_division=0),
        "recall_severe": recall_score(y_true, y_pred, zero_division=0),
        "f1_severe": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan

    conf = confusion_matrix(y_true, y_pred, labels=[1, 0])

    metrics["tp_severe"] = conf[0, 0]
    metrics["fn_severe"] = conf[0, 1]
    metrics["fp_severe"] = conf[1, 0]
    metrics["tn_not_severe"] = conf[1, 1]

    return metrics


def save_validation_accuracy_outputs(validation_pred):
    """
    Save fixed-threshold accuracy outputs for the independent validation set.
    """
    if TRUE_BINARY_COL not in validation_pred.columns:
        raise RuntimeError(f"Validation file is missing `{TRUE_BINARY_COL}`.")

    validation_pred[TRUE_BINARY_COL] = clean_numeric_keep_zero(
        validation_pred[TRUE_BINARY_COL]
    )

    d = validation_pred.dropna(subset=[TRUE_BINARY_COL, PRED_BINARY_COL]).copy()

    if len(d) == 0:
        raise RuntimeError("No validation rows have both true labels and predictions.")

    y_true = d[TRUE_BINARY_COL].astype(int)
    y_pred = d[PRED_BINARY_COL].astype(int)
    y_prob = d[PROB_COL].astype(float)

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics["severe_decline_threshold_used_for_validation"] = SEVERE_DECLINE_THRESHOLD
    metrics["prob_threshold_used"] = PROB_THRESHOLD

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    conf = confusion_matrix(y_true, y_pred, labels=[1, 0])

    conf_df = pd.DataFrame(
        conf,
        index=["actual_Severe decline", "actual_Not severe decline"],
        columns=["pred_Severe decline", "pred_Not severe decline"]
    )

    conf_df.to_csv(OUT_CONFUSION, encoding="utf-8-sig")

    report = classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=["Severe decline", "Not severe decline"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T.reset_index()

    report_df.to_csv(
        OUT_CLASS_REPORT,
        index=False,
        encoding="utf-8-sig"
    )

    return metrics_df, conf_df, report_df



# =====================================================
# SPATIAL OUTPUTS
# =====================================================

def make_join_key(series):
    """
    Create a conservative normalized settlement-name join key.

    The original `sett` field is preserved. This key only handles simple
    case, whitespace, and BOM/encoding artifacts.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )


def prepare_predictions_for_spatial_join(predictions):
    """
    Prepare the prediction table for joining to spatial layers.
    """
    pred = predictions.copy()

    if SETT_COL not in pred.columns:
        raise RuntimeError(f"Prediction table is missing `{SETT_COL}`.")

    pred[SETT_COL] = pred[SETT_COL].astype(str).str.strip()
    pred["_join_key"] = make_join_key(pred[SETT_COL])

    duplicate_keys = int(pred["_join_key"].duplicated().sum())

    if duplicate_keys > 0:
        print(
            f"[WARNING] Prediction table has {duplicate_keys} duplicate "
            "settlement join keys. Keeping the first prediction per key."
        )
        pred = pred.drop_duplicates(subset=["_join_key"], keep="first").copy()

    return pred


def write_prediction_spatial_layer(
    spatial_file,
    predictions,
    out_gpkg,
    layer_name,
    spatial_label
):
    """
    Join prediction attributes to a spatial layer and write a GeoPackage.

    The spatial layer is retained as the base layer. Therefore, every geometry
    in the input shapefile is preserved, and prediction attributes are attached
    where settlement names match.
    """
    print(f"[INFO] Reading {spatial_label} spatial layer...")
    gdf = gpd.read_file(spatial_file)
    gdf = clean_columns(gdf)

    if SETT_COL not in gdf.columns:
        raise RuntimeError(
            f"{spatial_label} spatial layer is missing `{SETT_COL}`.\n"
            f"Available columns:\n{gdf.columns.tolist()}"
        )

    pred = prepare_predictions_for_spatial_join(predictions)

    gdf[SETT_COL] = gdf[SETT_COL].astype(str).str.strip()
    gdf["_join_key"] = make_join_key(gdf[SETT_COL])

    core_pred_cols = [
        "_join_key",
        PROB_COL,
        PRED_BINARY_COL,
        PRED_CLASS_COL,
        "prediction_status",
        "prediction_source",
    ] + PREDICTORS + [f"logit_input_{p}" for p in PREDICTORS]

    # Add all non-overlapping prediction columns as well.
    extra_pred_cols = [
        c for c in pred.columns
        if c not in gdf.columns and c not in core_pred_cols
    ]

    pred_cols_to_join = [
        c for c in dict.fromkeys(core_pred_cols + extra_pred_cols)
        if c in pred.columns
    ]

    joined = gdf.merge(
        pred[pred_cols_to_join],
        on="_join_key",
        how="left",
        validate="many_to_one"
    )

    joined["prediction_join_matched"] = joined[PROB_COL].notna()

    joined.to_file(
        out_gpkg,
        layer=layer_name,
        driver="GPKG"
    )

    spatial_keys = set(gdf["_join_key"].dropna())
    pred_keys = set(pred["_join_key"].dropna())

    summary = {
        "spatial_layer": spatial_label,
        "input_spatial_file": str(spatial_file),
        "output_gpkg": str(out_gpkg),
        "output_layer": layer_name,
        "n_spatial_features": len(gdf),
        "n_spatial_features_with_prediction": int(joined["prediction_join_matched"].sum()),
        "n_spatial_features_without_prediction": int((~joined["prediction_join_matched"]).sum()),
        "n_unique_prediction_keys": len(pred_keys),
        "n_prediction_keys_matched_to_spatial": len(pred_keys.intersection(spatial_keys)),
        "n_prediction_keys_unmatched_to_spatial": len(pred_keys.difference(spatial_keys)),
    }

    print(f"[OK] Saved {spatial_label} GeoPackage:")
    print(f"  {out_gpkg}")
    print(f"  layer = {layer_name}")
    print(
        f"[INFO] {spatial_label}: matched "
        f"{summary['n_spatial_features_with_prediction']:,} of "
        f"{summary['n_spatial_features']:,} spatial features to predictions."
    )

    return summary


def write_spatial_prediction_outputs(predictions):
    """
    Write point and buffer GeoPackages with logistic prediction attributes.
    """
    print("[INFO] Writing spatial prediction GeoPackages...")

    summaries = []

    summaries.append(
        write_prediction_spatial_layer(
            spatial_file=SETT_POINTS_FILE,
            predictions=predictions,
            out_gpkg=OUT_POINTS_GPKG,
            layer_name="settlement_points_logistic_predictions",
            spatial_label="settlement_points"
        )
    )

    summaries.append(
        write_prediction_spatial_layer(
            spatial_file=SETT_BUFFERS_FILE,
            predictions=predictions,
            out_gpkg=OUT_BUFFERS_GPKG,
            layer_name="settlement_buffers_logistic_predictions",
            spatial_label="settlement_buffers"
        )
    )

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUT_SPATIAL_JOIN_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[OK] Saved spatial join summary:")
    print(f"  {OUT_SPATIAL_JOIN_SUMMARY}")

    return summary_df



# =====================================================
# MAIN
# =====================================================

def main():
    print("[INFO] Loading saved logistic model weights...")
    model = load_model_weights(MODEL_WEIGHTS_FILE)

    scaler_params = pd.DataFrame({
        "predictor": PREDICTORS,
        "training_mean": [model["scaler_mean"][p] for p in PREDICTORS],
        "training_std": [model["scaler_scale"][p] for p in PREDICTORS],
        "coefficient": [model["coefficients"][p] for p in PREDICTORS],
        "intercept": model["intercept"],
    })

    scaler_params.to_csv(
        OUT_SCALER_PARAMS,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # Predict all settlements
    # -----------------------------------------------------
    print("[INFO] Reading all-settlement prediction data...")
    pred_df = pd.read_csv(PREDICT_FILE)
    pred_df = clean_columns(pred_df)

    check_required_columns(
        pred_df,
        [SETT_COL] + PREDICTORS,
        "all-settlement prediction file"
    )

    print("[INFO] Applying model to ALL settlements...")
    all_predictions, all_predictor_map, all_valid_mask = apply_logistic_model(
        pred_df,
        model,
        source_label="all_settlements"
    )

    all_predictions = order_prediction_columns(all_predictions)

    all_predictions.to_csv(
        OUT_ALL_PREDICTIONS,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # Write spatial prediction outputs
    # -----------------------------------------------------
    spatial_join_summary = write_spatial_prediction_outputs(all_predictions)

    # -----------------------------------------------------
    # Predict master validation rows and calculate accuracy
    # -----------------------------------------------------
    print("[INFO] Reading master matched validation file...")
    master_df = pd.read_csv(MASTER_MATCHED_FILE)
    master_df = clean_columns(master_df)

    print("[INFO] Recalculating validation true labels using the training threshold...")
    master_df = add_validation_true_labels(master_df)

    print("[INFO] Applying model to master validation rows...")
    validation_predictions, validation_predictor_map, validation_valid_mask = apply_logistic_model(
        master_df,
        model,
        source_label="master_validation"
    )

    validation_predictions = order_prediction_columns(validation_predictions)

    validation_predictions.to_csv(
        OUT_VALIDATION_PREDICTIONS,
        index=False,
        encoding="utf-8-sig"
    )

    print("[INFO] Calculating fixed-threshold validation accuracy...")
    metrics_df, conf_df, report_df = save_validation_accuracy_outputs(
        validation_predictions
    )

    validation_rows_used = validation_predictions.dropna(
        subset=[TRUE_BINARY_COL, PRED_BINARY_COL]
    )

    run_summary = pd.DataFrame([{
        "severe_decline_threshold": SEVERE_DECLINE_THRESHOLD,
        "prob_threshold": PROB_THRESHOLD,
        "predictors": ", ".join(PREDICTORS),
        "model_weights_file": str(MODEL_WEIGHTS_FILE),
        "training_rows_used": model["n_train"],
        "training_severe_count": model["n_severe"],
        "training_not_severe_count": model["n_not_severe"],
        "all_settlement_rows_total": len(all_predictions),
        "all_settlement_rows_predicted": int((all_predictions["prediction_status"] == "predicted").sum()),
        "all_settlement_rows_missing_predictors": int((all_predictions["prediction_status"] != "predicted").sum()),
        "validation_rows_total": len(validation_predictions),
        "validation_rows_with_percent_decrease": int(validation_predictions[VALIDATION_PERCENT_CHANGE_COL].notna().sum()),
        "validation_true_severe_count": int((validation_predictions[TRUE_BINARY_COL] == 1).sum()),
        "validation_true_not_severe_count": int((validation_predictions[TRUE_BINARY_COL] == 0).sum()),
        "validation_rows_predicted": int((validation_predictions["prediction_status"] == "predicted").sum()),
        "validation_rows_used_for_metrics": len(validation_rows_used),
        "validation_label_source_column": VALIDATION_PERCENT_CHANGE_COL,
        "all_prediction_file": str(OUT_ALL_PREDICTIONS),
        "points_gpkg": str(OUT_POINTS_GPKG),
        "buffers_gpkg": str(OUT_BUFFERS_GPKG),
        "spatial_join_summary_file": str(OUT_SPATIAL_JOIN_SUMMARY),
        "validation_prediction_file": str(OUT_VALIDATION_PREDICTIONS),
        "metrics_file": str(OUT_METRICS),
        "confusion_matrix_file": str(OUT_CONFUSION),
        "classification_report_file": str(OUT_CLASS_REPORT),
    }])

    run_summary.to_csv(
        OUT_RUN_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------
    print("\n[MODEL PREDICTORS]")
    for p in PREDICTORS:
        print(f"  - {p}")

    print("\n[TRAINING LABEL THRESHOLD]")
    print(f"  {TRAIN_POP_CHANGE_COL} <= {SEVERE_DECLINE_THRESHOLD}")

    print("\n[VALIDATION LABEL THRESHOLD]")
    print(f"  {VALIDATION_PERCENT_CHANGE_COL} <= {SEVERE_DECLINE_THRESHOLD}")

    print("\n[PROBABILITY THRESHOLD]")
    print(f"  prob_severe_decline >= {PROB_THRESHOLD}")

    print("\n[TRAINING SAMPLE (from saved model)]")
    print(f"  Rows used for training: {model['n_train']}")
    print(f"  Severe decline:         {model['n_severe']}")
    print(f"  Not severe decline:     {model['n_not_severe']}")

    print("\n[VALIDATION TRUE CLASS COUNTS]")
    print(
        validation_predictions[TRUE_CLASS_COL]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n[ALL-SETTLEMENT PREDICTION SAMPLE]")
    print(f"  Total rows: {len(all_predictions)}")
    print(f"  Rows predicted: {(all_predictions['prediction_status'] == 'predicted').sum()}")
    print(f"  Rows missing predictors: {(all_predictions['prediction_status'] != 'predicted').sum()}")

    print("\n[ALL-SETTLEMENT PREDICTED CLASS COUNTS]")
    print(all_predictions[PRED_CLASS_COL].value_counts(dropna=False).to_string())

    print("\n[VALIDATION ACCURACY METRICS]")
    print(metrics_df.to_string(index=False))

    print("\n[VALIDATION CONFUSION MATRIX]")
    print(conf_df.to_string())

    print("\n[VALIDATION CLASSIFICATION REPORT]")
    print(report_df.to_string(index=False))

    print("\n[OK] Saved all-settlement predictions:")
    print(f"  {OUT_ALL_PREDICTIONS}")

    print("\n[OK] Saved point GeoPackage:")
    print(f"  {OUT_POINTS_GPKG}")

    print("\n[OK] Saved buffer GeoPackage:")
    print(f"  {OUT_BUFFERS_GPKG}")

    print("\n[OK] Saved spatial join summary:")
    print(f"  {OUT_SPATIAL_JOIN_SUMMARY}")

    print("\n[OK] Saved validation predictions:")
    print(f"  {OUT_VALIDATION_PREDICTIONS}")

    print("\n[OK] Saved validation metrics:")
    print(f"  {OUT_METRICS}")

    print("\n[OK] Saved validation confusion matrix:")
    print(f"  {OUT_CONFUSION}")

    print("\n[OK] Saved validation classification report:")
    print(f"  {OUT_CLASS_REPORT}")

    print("\n[OK] Saved run summary:")
    print(f"  {OUT_RUN_SUMMARY}")

    print("\n[INFO] Model weights loaded from:")
    print(f"  {MODEL_WEIGHTS_FILE}")

    print("\n[OK] Saved scaler parameters:")
    print(f"  {OUT_SCALER_PARAMS}")


if __name__ == "__main__":
    main()
