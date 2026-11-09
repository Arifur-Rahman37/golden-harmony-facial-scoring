"""
update_propsym_498.py
Makes the Proportion+Symmetry rows of the results CSVs consistent with the
498-image analysis (PropSym_image_level_FIXED498.csv / _subject_level_FIXED498.csv).
Run in the folder that contains the original CSV files:  python update_propsym_498.py
Writes corrected copies into ./corrected_498/ (originals are not modified).
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

OUT = "corrected_498"; os.makedirs(OUT, exist_ok=True)
FEATS = [f"P{i}-Score" for i in range(1, 5)] + [f"S{i}-Score" for i in range(1, 7)]

img = pd.read_csv("PropSym_image_level_FIXED498.csv")
sub = pd.read_csv("PropSym_subject_level_FIXED498.csv")
assert len(img) == 498 and len(sub) == 100
y = img["Human-Rating-Avg"].values; X = img[FEATS].values
naive = img["Combined-Geometric-Score-1to10"].values

# image-level naive
r_n, p_n = stats.pearsonr(naive, y); rho_n, rho_p = stats.spearmanr(naive, y)
# 5-fold CV OLS (same settings as regression_per_module.py: shuffle, random_state=42)
oof = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
    oof[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
r_cv, p_cv = stats.pearsonr(oof, y)
# subject-level naive
s_r, s_p = stats.pearsonr(sub["Mean-Combined-Geometric-Score-1to10"], sub["Mean-Human-Rating"])
s_rho, s_rho_p = stats.spearmanr(sub["Mean-Combined-Geometric-Score-1to10"], sub["Mean-Human-Rating"])
print(f"P+S image naive r={r_n:.4f} p={p_n:.4f} | CV OLS r={r_cv:.4f} p={p_cv:.4f} | subject r={s_r:.4f} p={s_p:.4f}")

PS = "Proportion + Symmetry (P1-P4, S1-S6)"

# 1. Correlation_Results_Summary.csv
c = pd.read_csv("Correlation_Results_Summary.csv")
m = c["Module"] == PS
c.loc[m & (c.Level == "Image-level"), ["N", "Pearson_r", "Pearson_p", "Spearman_r", "Spearman_p", "Source_File"]] = [498, round(r_n, 4), p_n, round(rho_n, 4), rho_p, "PropSym_image_level_FIXED498.csv"]
c.loc[m & (c.Level == "Subject-level"), ["N", "Pearson_r", "Pearson_p", "Spearman_r", "Spearman_p", "Source_File"]] = [100, round(s_r, 4), s_p, round(s_rho, 4), s_rho_p, "PropSym_subject_level_FIXED498.csv"]
c.to_csv(f"{OUT}/Correlation_Results_Summary.csv", index=False)

# 2./3. Paper_Results_Table_*.csv
for fn in ["Paper_Results_Table_Image_Level.csv", "Paper_Results_Table_Naive_vs_Regression.csv"]:
    t = pd.read_csv(fn); m = t["Module"] == PS
    t.loc[m & t.Method.str.startswith("Naive"), ["Pearson_r", "p_value", "N"]] = [round(r_n, 4), round(p_n, 4), 498]
    t.loc[m & t.Method.str.startswith("Linear"), ["Pearson_r", "p_value", "N"]] = [round(r_cv, 4), round(p_cv, 4), 498]
    t.to_csv(f"{OUT}/{fn}", index=False)

# 4. Paper_Results_Table_Subject_Level.csv
t = pd.read_csv("Paper_Results_Table_Subject_Level.csv"); m = (t["Module"] == PS)
t.loc[m, ["Pearson_r", "p_value", "N"]] = [round(s_r, 4), round(s_p, 4), 100]
t.to_csv(f"{OUT}/Paper_Results_Table_Subject_Level.csv", index=False)

# 5. Descriptive stats (P+S rows on 498 images / 100 subjects)
h_img = pd.Series(y); ai_img = pd.Series(naive)
d = pd.read_csv("Descriptive_Stats_Image_Level.csv")
d.loc[d.Score == "Proportion+Symmetry — AI score", ["N", "Mean", "SD", "Min", "Max"]] = [498, round(ai_img.mean(), 3), round(ai_img.std(), 3), round(ai_img.min(), 3), round(ai_img.max(), 3)]
d.loc[d.Score == "Proportion+Symmetry — Human rating", ["N", "Mean", "SD", "Min", "Max"]] = [498, round(h_img.mean(), 3), round(h_img.std(), 3), round(h_img.min(), 3), round(h_img.max(), 3)]
d.to_csv(f"{OUT}/Descriptive_Stats_Image_Level.csv", index=False)
a, h = sub["Mean-Combined-Geometric-Score-1to10"], sub["Mean-Human-Rating"]
d = pd.read_csv("Descriptive_Stats_Subject_Level.csv")
d.loc[d.Score == "Proportion+Symmetry — AI score", ["N", "Mean", "SD", "Min", "Max"]] = [100, round(a.mean(), 3), round(a.std(), 3), round(a.min(), 3), round(a.max(), 3)]
d.loc[d.Score == "Proportion+Symmetry — Human rating", ["N", "Mean", "SD", "Min", "Max"]] = [100, round(h.mean(), 3), round(h.std(), 3), round(h.min(), 3), round(h.max(), 3)]
d.to_csv(f"{OUT}/Descriptive_Stats_Subject_Level.csv", index=False)

# 6. Headline CIs
ci = pd.read_csv("Headline_Correlation_CIs.csv")
def pci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
i = ci["Comparison"] == "Proportion+Symmetry, regression CV, image-level"
lo, hi = pci(round(r_cv, 3), 498)
ci.loc[i, ["r", "n", "CI_low", "CI_high"]] = [round(r_cv, 3), 498, round(lo, 3), round(hi, 3)]
ci.to_csv(f"{OUT}/Headline_Correlation_CIs.csv", index=False)

# 7. regularized-regression JSON: OLS value was hard-coded (-0.066, from the 500-image run)
j = json.load(open("PropSym_regularized_regression.json"))
j["ols_regression_cv_r"] = round(float(r_cv), 4); j["naive_r"] = round(float(r_n), 4)
json.dump(j, open(f"{OUT}/PropSym_regularized_regression.json", "w"), indent=2)
print("written to", OUT)
