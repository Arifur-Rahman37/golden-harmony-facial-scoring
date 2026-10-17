"""
correlation_analysis.py
========================
Computes Pearson and Spearman correlation between AI (geometric) scores and
human ratings, at BOTH the image level and the subject level, for all three
modules of the Golden Harmony project:

    1. Golden Ratio only      (10Core output)
    2. Proportion + Symmetry  (PropSym output)
    3. Combined (20-feature, weighted-regression) score

This is meant to be run locally, with the actual output CSV files produced
by the three analysis scripts. It is self-contained and only needs pandas
and scipy.

INSTALL (one time):
    pip install pandas scipy --break-system-packages
    (drop --break-system-packages on Windows / normal venvs)

USAGE:
    1. Put this script in the same folder as your output CSV files
       (or edit the file paths in the MODULES list below).
    2. Run:
           python correlation_analysis.py
    3. It prints a results table to the screen AND saves it to
       "Correlation_Results_Summary.csv" in the same folder.

You can safely re-run this any time you regenerate the output CSVs — it
always reads fresh from disk, nothing is hardcoded from a previous run.
"""

import os
import sys
import pandas as pd
from scipy import stats

# =========================================================
# CONFIG — edit these paths/column names if your filenames differ
# =========================================================
# Each entry describes one module:
#   name              -> label used in the results table
#   image_csv         -> the *_image_level_output.csv file for that module
#   image_score_col   -> column name holding the AI score at image level
#   image_human_col   -> column name holding the human rating at image level
#   subject_csv       -> the *_subject_level_summary.csv file for that module
#   subject_score_col -> column name holding the AI score at subject level
#   subject_human_col -> column name holding the human rating at subject level
MODULES = [
    {
        "name": "Golden Ratio (R1-R10 only)",
        "image_csv": "Golden_10Core_image_level_output.csv",
        "image_score_col": "Golden-Rating-1to10",
        "image_human_col": "Human-Rating-Avg",
        "subject_csv": "Golden_10Core_subject_level_summary.csv",
        "subject_score_col": "Mean-Golden-Rating",
        "subject_human_col": "Mean-Human-Rating",
    },
    {
        "name": "Proportion + Symmetry (P1-P4, S1-S6)",
        "image_csv": "PropSym_image_level_output.csv",
        "image_score_col": "Combined-Geometric-Score-1to10",
        "image_human_col": "Human-Rating-Avg",
        "subject_csv": "PropSym_subject_level_summary.csv",
        "subject_score_col": "Mean-Combined-Geometric-Score-1to10",
        "subject_human_col": "Mean-Human-Rating",
    },
    {
        "name": "Combined (20-feature, weighted-regression)",
        "image_csv": "Golden_20Core_Combined_image_level_output.csv",
        "image_score_col": "Combined-Geometric-Score-1to10",
        "image_human_col": "Human-Rating-Avg",
        "subject_csv": "Golden_20Core_Combined_subject_level_summary.csv",
        "subject_score_col": "Mean-Combined-Geometric-Score-1to10",
        "subject_human_col": "Mean-Human-Rating",
    },
]

OUTPUT_SUMMARY_CSV = "Correlation_Results_Summary.csv"


# =========================================================
# Core correlation routine
# =========================================================
def compute_correlation(csv_path, score_col, human_col, level_label, module_name):
    """
    Reads one CSV file, drops rows where either the AI score or the human
    rating is missing, and returns a dict with Pearson r, Spearman r,
    their p-values, and the sample size actually used (n).
    """
    if not os.path.exists(csv_path):
        print(f"  [SKIPPED] File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if score_col not in df.columns:
        print(f"  [SKIPPED] Column '{score_col}' not found in {csv_path}")
        print(f"            Available columns: {list(df.columns)}")
        return None
    if human_col not in df.columns:
        print(f"  [SKIPPED] Column '{human_col}' not found in {csv_path}")
        print(f"            Available columns: {list(df.columns)}")
        return None

    clean = df[[score_col, human_col]].dropna()
    n = len(clean)

    if n < 3:
        print(f"  [SKIPPED] Not enough valid rows (n={n}) in {csv_path}")
        return None

    pearson_r, pearson_p = stats.pearsonr(clean[score_col], clean[human_col])
    spearman_r, spearman_p = stats.spearmanr(clean[score_col], clean[human_col])

    result = {
        "Module": module_name,
        "Level": level_label,
        "N": n,
        "Pearson_r": round(pearson_r, 4),
        "Pearson_p": pearson_p,
        "Spearman_r": round(spearman_r, 4),
        "Spearman_p": spearman_p,
        "Source_File": csv_path,
    }
    return result


def format_p(p):
    """Small helper so very small p-values print nicely instead of 0.0."""
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


# =========================================================
# Main
# =========================================================
def main():
    all_results = []

    for module in MODULES:
        print(f"\n=== {module['name']} ===")

        img_result = compute_correlation(
            module["image_csv"], module["image_score_col"], module["image_human_col"],
            level_label="Image-level", module_name=module["name"],
        )
        if img_result:
            all_results.append(img_result)
            print(
                f"  Image-level   (n={img_result['N']}): "
                f"Pearson r = {img_result['Pearson_r']} (p={format_p(img_result['Pearson_p'])}), "
                f"Spearman r = {img_result['Spearman_r']} (p={format_p(img_result['Spearman_p'])})"
            )

        subj_result = compute_correlation(
            module["subject_csv"], module["subject_score_col"], module["subject_human_col"],
            level_label="Subject-level", module_name=module["name"],
        )
        if subj_result:
            all_results.append(subj_result)
            print(
                f"  Subject-level (n={subj_result['N']}): "
                f"Pearson r = {subj_result['Pearson_r']} (p={format_p(subj_result['Pearson_p'])}), "
                f"Spearman r = {subj_result['Spearman_r']} (p={format_p(subj_result['Spearman_p'])})"
            )

    if not all_results:
        print("\nNo results were computed. Check that the CSV files exist in this folder")
        print("and that the column names in the MODULES config match your files.")
        sys.exit(1)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary_df[["Module", "Level", "N", "Pearson_r", "Spearman_r"]].to_string(index=False))
    print(f"\nSaved full results (with p-values) to: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
