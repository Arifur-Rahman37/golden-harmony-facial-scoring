# Golden Harmony: Automated Golden Ratio, Proportion, and Symmetry Facial Attractiveness Scoring

Code and data accompanying the paper **"Golden Ratio, Proportion, and Symmetry: A Calibrated Attractiveness Model."**

This repository reproduces the full analysis pipeline: facial-landmark-based feature extraction (Golden Ratio R1–R10, Proportion+Symmetry P1–P4/S1–S6, Combined 20-feature), human-rating calibration, naive vs. regression-weighted scoring, and all figures/tables used in the paper.

## Repository structure

```
scripts/
  integrated_facial_10_core_ratios_workflow.py   # Golden Ratio module (R1-R10): face crop, landmark detection, ratio scoring
  proportion_and_symmetry.py                     # Proportion+Symmetry module (P1-P4, S1-S6)
  Combined_Ratio_final.py                        # Combined 20-feature module
  calibrate_human_ratings.py                     # Per-rater z-score standardization (bias correction)
  correlation_analysis.py                        # Pearson correlation, naive vs. regression comparison
  regression_per_module.py                       # 5-fold cross-validated linear regression per module
  make_paper_charts.py                           # Generates Figures 1-5 for the paper
  make_result_plots.py                           # Additional result plots / diagnostics
  extra_stats.py                                 # ICC, confidence intervals, multiple-comparison correction, ridge/lasso regularization

data/processed/
  Calibrated_Human_Ratings.csv                   # Rater-bias-corrected human attractiveness ratings
  Golden_image_level.csv / Golden_subject_level.csv
  Golden_Regression_image_level_v2.csv / Golden_Regression_subject_level_v2.csv
  PropSym_image_level_FIXED498.csv / PropSym_subject_level_FIXED498.csv
  Combined_image_level.csv / Combined_subject_level.csv
  Correlation_Results_Summary.csv
  Paper_Results_Table_Naive_vs_Regression.csv
  Descriptive_Stats_Image_Level.csv / Descriptive_Stats_Subject_Level.csv
  feature_correlations.csv
  regression_weights.json                        # Fitted regression coefficients (Golden Ratio & Combined)
  ICC_by_form.csv                                # Inter-rater reliability (ICC) per rating form
  Headline_Correlation_CIs.csv                   # 95% confidence intervals for headline correlations
  feature_correlations_corrected.csv             # Per-feature correlations with Bonferroni/FDR correction
  PropSym_regularized_regression.json            # Ridge/lasso regularized regression results (Proportion+Symmetry)

figures/
  fig1_scatter_image_level.png      # Figure 1: AI Combined score vs. human rating, image-level (n=498)
  fig2_scatter_subject_level.png    # Figure 2: AI Combined score vs. human rating, subject-level (n=100)
  fig3_naive_vs_regression_bar.png  # Figure 3: Naive vs. regression-weighted correlation by module
  fig4_distribution_comparison.png  # Figure 4: Distribution of AI scores vs. human ratings
  fig5_feature_correlation_bar.png  # Figure 5: Per-feature Pearson correlation (all 20 features)
```

## Pipeline overview

1. **Face cropping & landmark detection** (`integrated_facial_10_core_ratios_workflow.py`, `proportion_and_symmetry.py`) — MediaPipe FaceLandmarker extracts facial landmarks from each image; a golden-ratio-proportioned crop is derived from inter-eye distance, and landmarks are re-detected on the crop as a two-pass quality check.
2. **Feature scoring** — Each module computes its geometric ratios/measures and converts them to 1–10 scores via a Gaussian-decay formula against the golden ratio (or symmetry/proportion target).
3. **Human rating calibration** (`calibrate_human_ratings.py`) — Raw 1–10 ratings from 58 raters are corrected for individual rater leniency/severity bias via per-rater z-score standardization.
4. **Scoring aggregation & regression** (`regression_per_module.py`) — Compares naive equal-weight averaging against a 5-fold cross-validated linear regression that learns per-feature weights.
5. **Analysis & figures** (`correlation_analysis.py`, `make_paper_charts.py`, `make_result_plots.py`) — Computes Pearson correlations against calibrated human ratings and generates all paper figures/tables.

## Dataset

500 images (5 per subject) of 100 Bangladeshi male celebrities, each rated by 58 independent human raters (5 Google Forms, 100 images each). See the paper's Methods section for full details on image collection, rater demographics, and the two-pass quality-control exclusion (2 of 500 images excluded).

## A note on the images

This repository intentionally does **not** include the raw subject photographs or their derived crops/landmark overlays (`crop_output/`, `landmark_output/`, `viz_output/`). The dataset consists of internet-sourced photographs of public-figure celebrities, and we have not established redistribution rights for them. Only the extracted numeric features/scores (`data/processed/`) and the summary charts (`figures/`, which contain no faces) are shared, which is sufficient to reproduce every reported statistic and figure in the paper. The scripts in `scripts/` can be re-run on any equivalent local image folder to regenerate the same kind of output.

## Requirements

- Python 3.x
- `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`

## Citation

If you use this code or data, please cite the paper (citation details to be added upon publication).
