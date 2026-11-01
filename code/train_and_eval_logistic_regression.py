# -*- coding: utf-8 -*-

"""
logistic_regression_single_model.py
===================================

Fit and evaluate one two-predictor logistic regression model that predicts
whether an Arctic settlement experienced severe population decline from its
nighttime lights (NTL) change metrics.

The model is evaluated with stratified 5-fold cross-validation. A final
model is then fitted on the full dataset to report coefficients and odds
ratios.


INPUT
-----
    outputs/ntl_analysis_sett/model_training_dataset.csv

    One row per settlement. Required columns:
        sett                 - settlement identifier
        pop_decrease         - population change (%) for the settlement;
                               negative values are declines
        collapse_loss_pct    - % NTL loss from the 1992-1995 baseline to the
                               series trough (see
                               code/time_series/ntl_time_series_analysis.py)
        recent_vs_early_pct  - % NTL change from 1992-1995 to 2010-2013

    Byte-order marks (BOMs) and whitespace are stripped from column names.
    Numeric columns are coerced to numbers: text placeholders ("", "nan",
    "None", "NULL", "<NA>", ...) and +/-inf become NaN, while genuine zeros
    are kept. Rows missing pop_decrease or either predictor are dropped
    (complete-case analysis).


OUTCOME
-------
    severe_decline_binary = 1 if pop_decrease <= SEVERE_DECLINE_THRESHOLD
                            (-25, i.e. a population loss of 25% or more),
                            otherwise 0.

    severe_decline_class gives the same thing as a text label
    ("Severe decline" / "Not severe decline").

    The script stops with an error if only one class is present, or if
    the smaller class has fewer than N_SPLITS (5) settlements, because
    stratified 5-fold CV is then impossible.


METHOD
------
    1. Stratified 5-fold cross-validation
           Folds are shuffled with RANDOM_STATE = 42 for reproducibility and
           keep the severe/not-severe ratio roughly equal in every fold.
           In each fold:
             - Predictors are standardised (z-scores) with a StandardScaler
               fitted on the training folds only, then applied to the test
               fold, so no information leaks from the test data.
             - An intercept is added and a statsmodels Logit model is fitted
               (L-BFGS, up to 500 iterations).
             - Test settlements get a predicted probability of severe
               decline and are classed as severe when the probability is
               >= PROB_THRESHOLD (0.50).
             - Fold-level metrics and training-fit statistics (AIC, BIC,
               McFadden pseudo-R2) are recorded.

    2. Pooled out-of-fold evaluation
           Every settlement is predicted exactly once, by the model that did
           not see it in training. These out-of-fold predictions are pooled
           to compute overall metrics, a classification report and a
           confusion matrix.

    3. Final full-data model
           The predictors are standardised over the full dataset and the
           model is refitted on all settlements. Its coefficients are
           therefore per one standard deviation of each predictor, and
           odds_ratio = exp(coefficient) is the change in the odds of severe
           decline for a 1 SD increase in that predictor. This model is for
           interpretation; its performance is not separately evaluated.


METRICS
-------
    accuracy, balanced_accuracy, roc_auc
    precision / recall / f1 for the severe class (pos_label = 1)
    macro and weighted precision / recall / f1
    tp_severe, fn_severe, fp_severe, tn_not_severe (confusion matrix counts,
    with "severe" as the positive class)

    roc_auc is NaN for a fold whose test set contains only one class.


OUTPUTS
-------
    Written to outputs/ntl_analysis_sett/logistic_two_variable_cv/ (created
    if missing). CSVs are UTF-8 with BOM.

        two_variable_class_counts.csv
            Number of settlements in each outcome class after filtering.
        two_variable_cv_fold_results.csv
            One row per fold: train/test sizes, AIC, BIC, pseudo-R2 and all
            metrics above.
        two_variable_cv_predictions.csv
            One row per settlement: inputs, observed class, fold, out-of-
            fold probability and predicted class.
        two_variable_oof_cv_classification_report.csv
            sklearn classification report on the pooled out-of-fold
            predictions.
        two_variable_oof_cv_confusion_matrix.csv
            Pooled confusion matrix (rows = actual, columns = predicted).
        two_variable_full_model_coefficients.csv
            Full-data model terms: coefficient, odds ratio, standard error,
            z value and p value.
        two_variable_full_model_summary.txt
            The full statsmodels summary of the full-data model.

    The pooled out-of-fold metrics are printed to the console only; they
    are not saved to a file.


SETTINGS
--------
    SEVERE_DECLINE_THRESHOLD  population change (%) at or below which a
                              settlement counts as severe decline (-25)
    PROB_THRESHOLD            probability cut-off for predicting severe (0.50)
    N_SPLITS                  number of CV folds (5)
    RANDOM_STATE              seed for fold shuffling (42)
    PREDICTORS                model predictor columns
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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
# FILES
# =====================================================

IN_FILE = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\model_training_dataset.csv"
)

OUT_DIR = Path(
    r"D:\PhD_main\chapter_3\outputs\ntl_analysis_sett\logistic_two_variable_cv"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLASS_COUNTS = OUT_DIR / "two_variable_class_counts.csv"
OUT_CV_FOLDS = OUT_DIR / "two_variable_cv_fold_results.csv"
OUT_CV_PREDICTIONS = OUT_DIR / "two_variable_cv_predictions.csv"
OUT_CV_REPORT = OUT_DIR / "two_variable_oof_cv_classification_report.csv"
OUT_CV_CONFUSION = OUT_DIR / "two_variable_oof_cv_confusion_matrix.csv"
OUT_FULL_COEFS = OUT_DIR / "two_variable_full_model_coefficients.csv"
OUT_FULL_SUMMARY = OUT_DIR / "two_variable_full_model_summary.txt"

# =====================================================
# SETTINGS
# =====================================================

POP_COL = "pop_decrease"
BINARY_COL = "severe_decline_binary"
CLASS_COL = "severe_decline_class"

SEVERE_DECLINE_THRESHOLD = -25
PROB_THRESHOLD = 0.50

N_SPLITS = 5
RANDOM_STATE = 42

PREDICTORS = [
    "collapse_loss_pct",
    "recent_vs_early_pct",
]

# =====================================================
# HELPERS
# =====================================================

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace("ï»¿", "", regex=False)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def clean_numeric_keep_zero(series):
    s = series.copy()

    if s.dtype == object:
        s = (
            s.astype(str)
            .str.strip()
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


def fit_logit(X, y):
    model = sm.Logit(y, X)

    result = model.fit(
        method="lbfgs",
        disp=False,
        maxiter=500,
    )

    return result


def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "precision_severe": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_severe": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_severe": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),

        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

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


# =====================================================
# READ DATA
# =====================================================

print("[INFO] Reading data...")

df = pd.read_csv(IN_FILE)
df = clean_columns(df)

required = ["sett", POP_COL] + PREDICTORS
missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}\n"
        f"Available columns:\n{df.columns.tolist()}"
    )

df["sett"] = df["sett"].astype(str).str.strip()

for col in [POP_COL] + PREDICTORS:
    df[col] = clean_numeric_keep_zero(df[col])

df = df.dropna(subset=[POP_COL] + PREDICTORS).copy()

df[BINARY_COL] = np.where(
    df[POP_COL] <= SEVERE_DECLINE_THRESHOLD,
    1,
    0,
)

df[CLASS_COL] = np.where(
    df[BINARY_COL] == 1,
    "Severe decline",
    "Not severe decline",
)

class_counts = (
    df[CLASS_COL]
    .value_counts()
    .reindex(["Severe decline", "Not severe decline"])
    .reset_index()
)

class_counts.columns = [CLASS_COL, "n"]
class_counts.to_csv(OUT_CLASS_COUNTS, index=False, encoding="utf-8-sig")

print("\n[CLASS COUNTS]")
print(class_counts.to_string(index=False))

if df[BINARY_COL].nunique() < 2:
    raise RuntimeError("Only one class present after filtering.")

if df[BINARY_COL].value_counts().min() < N_SPLITS:
    raise RuntimeError(
        f"Minority class has fewer than {N_SPLITS} samples. "
        "Cannot run stratified 5-fold CV."
    )

# =====================================================
# 5-FOLD CROSS-VALIDATION
# =====================================================

print("\n[INFO] Running 5-fold CV...")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_rows = []
pred_rows = []

X_all = df[PREDICTORS].copy()
y_all = df[BINARY_COL].astype(int)

for fold_id, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):

    print(f"[INFO] Starting fold {fold_id}...")

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    X_train_raw = train[PREDICTORS]
    X_test_raw = test[PREDICTORS]

    y_train = train[BINARY_COL].astype(int)
    y_test = test[BINARY_COL].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train = pd.DataFrame(
        X_train_scaled,
        columns=PREDICTORS,
        index=train.index,
    )

    X_test = pd.DataFrame(
        X_test_scaled,
        columns=PREDICTORS,
        index=test.index,
    )

    X_train = sm.add_constant(X_train, has_constant="add")
    X_test = sm.add_constant(X_test, has_constant="add")

    result = fit_logit(X_train, y_train)

    pred_prob = result.predict(X_test)
    pred_binary = (pred_prob >= PROB_THRESHOLD).astype(int)

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=pred_binary,
        y_prob=pred_prob,
    )

    fold_row = {
        "fold": fold_id,
        "n_train": len(train),
        "n_test": len(test),
        "prob_threshold": PROB_THRESHOLD,
        "aic_train": result.aic,
        "bic_train": result.bic,
        "pseudo_r2_mcfadden_train": result.prsquared,
    }

    fold_row.update(metrics)
    fold_rows.append(fold_row)

    fold_pred = test[
        [
            "sett",
            POP_COL,
            CLASS_COL,
            BINARY_COL,
        ] + PREDICTORS
    ].copy()

    fold_pred["fold"] = fold_id
    fold_pred["cv_prob_severe_decline"] = pred_prob.values
    fold_pred["cv_pred_binary"] = pred_binary.values
    fold_pred["cv_pred_class"] = np.where(
        fold_pred["cv_pred_binary"] == 1,
        "Severe decline",
        "Not severe decline",
    )

    pred_rows.append(fold_pred)

    print(f"[INFO] Finished fold {fold_id}.")

# =====================================================
# SAVE CV OUTPUTS
# =====================================================

cv_folds = pd.DataFrame(fold_rows)
cv_preds = pd.concat(pred_rows, ignore_index=True)

cv_folds.to_csv(OUT_CV_FOLDS, index=False, encoding="utf-8-sig")
cv_preds.to_csv(OUT_CV_PREDICTIONS, index=False, encoding="utf-8-sig")

# =====================================================
# POOLED OUT-OF-FOLD CV PERFORMANCE
# =====================================================

y_cv = cv_preds[BINARY_COL].astype(int)
pred_cv = cv_preds["cv_pred_binary"].astype(int)
prob_cv = cv_preds["cv_prob_severe_decline"].astype(float)

pooled_metrics = calculate_metrics(y_cv, pred_cv, prob_cv)

report = classification_report(
    y_cv,
    pred_cv,
    labels=[1, 0],
    target_names=[
        "Severe decline",
        "Not severe decline",
    ],
    output_dict=True,
    zero_division=0,
)

report_df = (
    pd.DataFrame(report)
    .T
    .reset_index()
    .rename(columns={"index": "class_or_average"})
)

report_df.to_csv(OUT_CV_REPORT, index=False, encoding="utf-8-sig")

conf = confusion_matrix(y_cv, pred_cv, labels=[1, 0])

conf_df = pd.DataFrame(
    conf,
    index=[
        "actual_Severe decline",
        "actual_Not severe decline",
    ],
    columns=[
        "pred_Severe decline",
        "pred_Not severe decline",
    ],
)

conf_df.to_csv(OUT_CV_CONFUSION, encoding="utf-8-sig")

# =====================================================
# FIT FINAL MODEL ON FULL DATASET
# =====================================================

print("\n[INFO] Fitting final model on full dataset...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[PREDICTORS])

X_full = pd.DataFrame(
    X_scaled,
    columns=PREDICTORS,
    index=df.index,
)

X_full = sm.add_constant(X_full, has_constant="add")
y_full = df[BINARY_COL].astype(int)

full_result = fit_logit(X_full, y_full)

coef_rows = []

for term, coef in full_result.params.items():
    coef_rows.append({
        "term": term,
        "coefficient": coef,
        "odds_ratio": np.exp(coef),
        "std_error": full_result.bse.get(term, np.nan),
        "z_value": full_result.tvalues.get(term, np.nan),
        "p_value": full_result.pvalues.get(term, np.nan),
    })

coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(OUT_FULL_COEFS, index=False, encoding="utf-8-sig")

with open(OUT_FULL_SUMMARY, "w", encoding="utf-8") as f:
    f.write(full_result.summary().as_text())

# =====================================================
# PRINT RESULTS
# =====================================================

print("\n[POOLED OUT-OF-FOLD CV METRICS]")
for k, v in pooled_metrics.items():
    print(f"{k}: {v}")

print("\n[CV CLASSIFICATION REPORT]")
print(report_df.to_string(index=False))

print("\n[CV CONFUSION MATRIX]")
print(conf_df.to_string())

print("\n[FULL-DATA COEFFICIENTS]")
print(coef_df.to_string(index=False))

print("\n[OK] Saved outputs to:")
print(OUT_DIR)
