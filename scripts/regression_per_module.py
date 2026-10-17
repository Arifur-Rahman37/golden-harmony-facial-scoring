"""
regression_per_module.py
=========================
Fits a cross-validated linear regression (same method already used to build
the "Combined" weighted score) SEPARATELY for:

    1. Golden Ratio only      -> 10 features (R1-R10)
    2. Proportion + Symmetry  -> 10 features (P1-P4, S1-S6)

Goal: check whether the naive-average's weak correlation with human rating
(Golden r~0.08-0.14, PropSym r~ -0.04-0.06) is a problem of EQUAL WEIGHTING
(fixable by regression, like it was for Combined), or whether these smaller
feature sets just don't carry enough signal on their own (not fixable by
reweighting alone).

Method (identical to how the Combined weights were produced):
    - Linear Regression, 20 features -> here 10 features -> 1 target (human rating)
    - 5-fold cross-validation (KFold, shuffle=True, random_state=42)
    - Report OUT-OF-SAMPLE Pearson r (the honest number) AND the in-sample
      r from a model fit on all the data (the optimistic number), so both
      can be reported transparently like the Combined module.

INSTALL:
    pip install pandas numpy scikit-learn scipy --break-system-packages

USAGE:
    python regression_per_module.py
    (edit the two CONFIG blocks below if your technical CSV filenames /
    column names differ)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy import stats

# =========================================================
# CONFIG
# =========================================================
GOLDEN_CSV = "Golden_technical.csv"
GOLDEN_FEATURE_COLS = [f"R{i}-Phi-Closeness-Score" for i in range(1, 11)]
GOLDEN_HUMAN_COL = "Human-Rating-Avg"
GOLDEN_NAIVE_COL = "Golden-Rating-1to10"   # existing naive/equal-weight score, for comparison

PROPSYM_CSV = "PropSym_technical.csv"
PROPSYM_FEATURE_COLS = [f"P{i}-Score" for i in range(1, 5)] + [f"S{i}-Score" for i in range(1, 7)]
PROPSYM_HUMAN_COL = "Human-Rating-Avg"
PROPSYM_NAIVE_COL = "Combined-Geometric-Score-1to10"  # PropSym script's own naive combined score

N_FOLDS = 5
RANDOM_STATE = 42


# =========================================================
# Core routine
# =========================================================
def run_module(name, csv_path, feature_cols, human_col, naive_col):
    print(f"\n{'=' * 70}")
    print(f"MODULE: {name}")
    print(f"{'=' * 70}")

    df = pd.read_csv(csv_path)
    needed = feature_cols + [human_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  [SKIPPED] Missing columns in {csv_path}: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        return None

    data = df[feature_cols + [human_col] + ([naive_col] if naive_col in df.columns else [])].dropna()
    X = data[feature_cols].values
    y = data[human_col].values
    n = len(data)
    print(f"  Rows used (after dropping missing): n = {n}")

    # ---- Naive (existing equal-weight) score correlation, for comparison ----
    if naive_col in data.columns:
        r_naive, p_naive = stats.pearsonr(data[naive_col], y)
        print(f"  Naive/equal-weight score vs human rating : Pearson r = {r_naive:.4f} (p={p_naive:.4g})")

    # ---- Cross-validated regression (honest, out-of-sample) ----
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        oof_predictions[test_idx] = model.predict(X[test_idx])

    r_cv, p_cv = stats.pearsonr(oof_predictions, y)
    r2_cv = r_cv ** 2
    print(f"  Cross-validated regression (out-of-sample): Pearson r = {r_cv:.4f} (p={p_cv:.4g}), R^2 = {r2_cv:.4f}")

    # ---- Full-data fit (in-sample, optimistic) — these are the weights you'd embed in the script ----
    full_model = LinearRegression()
    full_model.fit(X, y)
    in_sample_pred = full_model.predict(X)
    r_in, p_in = stats.pearsonr(in_sample_pred, y)
    print(f"  Full-data fit (in-sample, for reference)  : Pearson r = {r_in:.4f} (p={p_in:.4g})")

    print(f"\n  Learned weights (full-data fit):")
    print(f"    Intercept: {full_model.intercept_}")
    for feat, coef in zip(feature_cols, full_model.coef_):
        print(f"    {feat}: {coef}")

    return {
        "module": name,
        "n": n,
        "r_naive": r_naive if naive_col in data.columns else None,
        "r_cv_out_of_sample": r_cv,
        "r_in_sample": r_in,
        "intercept": full_model.intercept_,
        "weights": dict(zip(feature_cols, full_model.coef_)),
    }


def main():
    golden_result = run_module(
        "Golden Ratio (R1-R10)", GOLDEN_CSV, GOLDEN_FEATURE_COLS, GOLDEN_HUMAN_COL, GOLDEN_NAIVE_COL
    )
    propsym_result = run_module(
        "Proportion + Symmetry (P1-P4, S1-S6)", PROPSYM_CSV, PROPSYM_FEATURE_COLS, PROPSYM_HUMAN_COL, PROPSYM_NAIVE_COL
    )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for res in [golden_result, propsym_result]:
        if res is None:
            continue
        print(
            f"{res['module']}: naive r={res['r_naive']:.4f} -> "
            f"regression (CV) r={res['r_cv_out_of_sample']:.4f} -> "
            f"regression (in-sample) r={res['r_in_sample']:.4f}  (n={res['n']})"
        )


if __name__ == "__main__":
    main()
