import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_predict
import json

# =====================================================================
# 1. INTER-RATER RELIABILITY (ICC) -- per form group (5 forms x ~100 imgs)
# =====================================================================
merged = pd.read_excel("Merged_Human_Ratings.xlsx")

icc_results = []
for form_num in range(1, 6):
    cols = [c for c in merged.columns if c.startswith(f"Form{form_num}_User")]
    sub = merged[["Image"] + cols].dropna(subset=cols, how="all")
    sub = sub[sub[cols].notna().any(axis=1)]
    # long format for pingouin ICC
    long = sub.melt(id_vars="Image", value_vars=cols, var_name="Rater", value_name="Rating").dropna()
    icc = pg.intraclass_corr(data=long, targets="Image", raters="Rater", ratings="Rating")
    icc2k = icc[icc["Type"] == "ICC(A,k)"].iloc[0]  # average-measures, two-way random, absolute agreement
    icc2_1 = icc[icc["Type"] == "ICC(A,1)"].iloc[0]  # single-rater version
    n_raters = len(cols)
    n_images = sub["Image"].nunique()
    icc_results.append({
        "Form": form_num,
        "N_raters": n_raters,
        "N_images": n_images,
        "ICC2_single": round(icc2_1["ICC"], 4),
        "ICC2_single_CI_low": round(icc2_1["CI95"][0], 4),
        "ICC2_single_CI_high": round(icc2_1["CI95"][1], 4),
        "ICC2k_avg": round(icc2k["ICC"], 4),
        "ICC2k_CI_low": round(icc2k["CI95"][0], 4),
        "ICC2k_CI_high": round(icc2k["CI95"][1], 4),
    })

icc_df = pd.DataFrame(icc_results)
print("=== ICC per form (two-way random, absolute agreement) ===")
print(icc_df.to_string(index=False))
icc_df.to_csv("ICC_by_form.csv", index=False)

overall_icc2_single = icc_df["ICC2_single"].mean()
overall_icc2k = icc_df["ICC2k_avg"].mean()
print(f"\nMean single-rater ICC(2,1) across 5 forms: {overall_icc2_single:.4f}")
print(f"Mean average-rater ICC(2,k) across 5 forms: {overall_icc2k:.4f}")

# =====================================================================
# 2. CONFIDENCE INTERVALS for headline Pearson r values (Fisher z)
# =====================================================================
def pearson_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    return np.tanh(lo), np.tanh(hi)

headline = [
    ("Golden Ratio, naive, image-level", 0.082, 498),
    ("Golden Ratio, regression CV, image-level", 0.322, 498),
    ("Golden Ratio, regression in-sample, image-level", 0.388, 498),
    ("Proportion+Symmetry, naive, image-level", 0.054, 498),
    ("Proportion+Symmetry, regression CV, image-level", -0.066, 498),
    ("Combined, naive, image-level", 0.091, 498),
    ("Combined, regression in-sample, image-level", 0.412, 498),
    ("Combined, regression CV, image-level", 0.295, 498),
    ("Combined, regression in-sample, subject-level", 0.556, 100),
]
print("\n=== 95% CI for headline correlations (Fisher z) ===")
ci_rows = []
for label, r, n in headline:
    lo, hi = pearson_ci(r, n)
    print(f"{label}: r={r:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], n={n}")
    ci_rows.append({"Comparison": label, "r": r, "n": n, "CI_low": round(lo, 3), "CI_high": round(hi, 3)})
pd.DataFrame(ci_rows).to_csv("Headline_Correlation_CIs.csv", index=False)

# =====================================================================
# 3. MULTIPLE COMPARISON CORRECTION on the 20 per-feature p-values
# =====================================================================
from statsmodels.stats.multitest import multipletests

fc = pd.read_csv("feature_correlations.csv")
reject_bonf, p_bonf, _, _ = multipletests(fc["p"], alpha=0.05, method="bonferroni")
reject_fdr, p_fdr, _, _ = multipletests(fc["p"], alpha=0.05, method="fdr_bh")
fc["p_bonferroni"] = p_bonf
fc["sig_bonferroni_0.05"] = reject_bonf
fc["p_fdr_bh"] = p_fdr
fc["sig_fdr_0.05"] = reject_fdr
fc.to_csv("feature_correlations_corrected.csv", index=False)
print("\n=== Multiple comparison correction (20 features) ===")
print(fc[["Feature", "r", "p", "p_bonferroni", "sig_bonferroni_0.05", "p_fdr_bh", "sig_fdr_0.05"]].to_string(index=False))
print(f"\nSignificant at raw p<0.05: {(fc['p']<0.05).sum()}/20")
print(f"Significant after Bonferroni (p<0.05): {reject_bonf.sum()}/20")
print(f"Significant after FDR (Benjamini-Hochberg, q<0.05): {reject_fdr.sum()}/20")

# =====================================================================
# 4. RIDGE / LASSO regularized regression for Proportion+Symmetry (10 feat)
# =====================================================================
ps = pd.read_csv("PropSym_image_level_FIXED498.csv")
feat_cols = ["P1-Score", "P2-Score", "P3-Score", "P4-Score", "S1-Score", "S2-Score", "S3-Score", "S4-Score", "S5-Score", "S6-Score"]
X = ps[feat_cols].values
y = ps["Human-Rating-Avg"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
ridge_pred = cross_val_predict(ridge, X, y, cv=kf)
ridge_r, ridge_p = stats.pearsonr(ridge_pred, y)

lasso = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=20000)
lasso_pred = cross_val_predict(lasso, X, y, cv=kf)
lasso_r, lasso_p = stats.pearsonr(lasso_pred, y)

# fit on full data to report chosen alpha / coefficients
ridge.fit(X, y)
lasso.fit(X, y)

print("\n=== Proportion+Symmetry: regularized regression (5-fold CV, out-of-sample) ===")
print(f"Plain OLS regression (already reported):      r = -0.066")
print(f"Ridge regression  (alpha={ridge.alpha_:.3f}):           r = {ridge_r:.3f}, p = {ridge_p:.4f}")
print(f"Lasso regression  (alpha={lasso.alpha_:.4f}):          r = {lasso_r:.3f}, p = {lasso_p:.4f}")
print(f"Naive equal-weight average (already reported): r = 0.054")

reg_results = {
    "naive_r": 0.054,
    "ols_regression_cv_r": -0.066,
    "ridge_cv_r": round(float(ridge_r), 4),
    "ridge_cv_p": float(ridge_p),
    "ridge_alpha": float(ridge.alpha_),
    "ridge_coefficients": dict(zip(feat_cols, [round(float(c), 4) for c in ridge.coef_])),
    "lasso_cv_r": round(float(lasso_r), 4),
    "lasso_cv_p": float(lasso_p),
    "lasso_alpha": float(lasso.alpha_),
    "lasso_coefficients": dict(zip(feat_cols, [round(float(c), 4) for c in lasso.coef_])),
}
with open("PropSym_regularized_regression.json", "w") as f:
    json.dump(reg_results, f, indent=2)

print("\nLasso selected (non-zero) features:", [k for k, v in reg_results["lasso_coefficients"].items() if v != 0])
print("\nAll outputs saved: ICC_by_form.csv, Headline_Correlation_CIs.csv, feature_correlations_corrected.csv, PropSym_regularized_regression.json")
