"""
make_result_plots.py
=====================
Generates the Results-section figures for the Golden Harmony Scoring paper:

  Fig 1. Image-level scatter: Combined AI score vs Human rating (n=498)
  Fig 2. Subject-level scatter: Combined AI score vs Human rating (n=100)
  Fig 3. Grouped bar: naive vs cross-validated-regression correlation,
         across Golden Ratio / Proportion+Symmetry / Combined
  Fig 4. Distribution comparison: AI score vs Human rating (image-level)
  Fig 5. Per-feature correlation with human rating (20 features, R1-R10/P1-P4/S1-S6)

Colors follow the validated dataviz reference palette:
  blue   #2a78d6  (primary / AI score)
  orange #eb6834  (secondary / naive, human rating)
  diverging blue<->red with gray midpoint (per-feature correlation sign)

Output: PNG files in /home/claude/work/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ---------------------------------------------------------------
# Palette (validated reference instance, light mode)
# ---------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
GRAY_MID = "#f0efec"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 11,
})

OUT_DIR = "/home/claude/work/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def style_axes(ax, hide_top_right=True):
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# =================================================================
# Fig 1 & 2: Scatter plots (image-level and subject-level)
# =================================================================
def make_scatter(csv_path, score_col, human_col, n_expected, title, fname, point_size=22):
    df = pd.read_csv(csv_path)[[score_col, human_col]].dropna()
    x = df[human_col].values
    y = df[score_col].values
    r, p = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
    ax.scatter(x, y, s=point_size, color=BLUE, alpha=0.55, edgecolors="none", zorder=3)

    # Best-fit line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=ORANGE, linewidth=2, zorder=4,
            label="Best-fit line")

    style_axes(ax)
    ax.set_xlabel("Human Rating (1–10)")
    ax.set_ylabel("AI Combined Score (1–10)")
    ax.set_title(title, fontsize=12.5, color=INK_PRIMARY, pad=12)

    p_str = "< 0.0001" if p < 0.0001 else f"= {p:.4f}"
    ax.text(0.03, 0.97, f"Pearson r = {r:.3f}\np {p_str}\nn = {len(df)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
            color=INK_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID))

    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {fname} | r={r:.4f}, p={p:.4g}, n={len(df)}")
    return r, p, len(df)


# =================================================================
# Fig 3: Grouped bar — naive vs CV-regression correlation per module
# =================================================================
def make_naive_vs_regression_bar():
    modules = ["Golden\nRatio", "Proportion +\nSymmetry", "Combined\n(20-feature)"]
    naive_r = [0.0820, 0.0546, 0.0918]     # image-level naive r for each module
    cv_r    = [0.3219, -0.0658, 0.2950]    # image-level cross-validated regression r

    x = np.arange(len(modules))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    bars1 = ax.bar(x - width/2, naive_r, width, color=ORANGE, label="Naive (equal-weight) average", zorder=3)
    bars2 = ax.bar(x + width/2, cv_r, width, color=BLUE, label="Regression-weighted (cross-validated)", zorder=3)

    ax.axhline(0, color=BASELINE, linewidth=1, zorder=2)
    style_axes(ax)
    ax.set_ylabel("Pearson r vs Human Rating")
    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.set_title("Naive Averaging vs Regression-Weighted Scoring\n(image-level correlation with human rating)",
                 fontsize=12.5, pad=12)

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 0.012 if h >= 0 else -0.012
            ax.text(b.get_x() + b.get_width()/2, h + offset, f"{h:.3f}",
                    ha="center", va=va, fontsize=9.5, color=INK_SECONDARY)

    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    ax.set_ylim(-0.12, 0.42)
    fig.tight_layout()
    fname = "fig3_naive_vs_regression_bar.png"
    fig.savefig(os.path.join(OUT_DIR, fname), facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {fname}")


# =================================================================
# Fig 4: Distribution comparison — AI score vs Human rating (image-level)
# =================================================================
def make_distribution_comparison(csv_path, score_col, human_col, fname):
    df = pd.read_csv(csv_path)[[score_col, human_col]].dropna()

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    bins = np.linspace(1, 10, 28)

    ax.hist(df[human_col], bins=bins, color=ORANGE, alpha=0.55, label="Human Rating", zorder=3)
    ax.hist(df[score_col], bins=bins, color=BLUE, alpha=0.55, label="AI Combined Score", zorder=3)

    style_axes(ax)
    ax.set_xlabel("Rating (1–10)")
    ax.set_ylabel("Number of Images")
    ax.set_title("Distribution of AI Scores vs Human Ratings (Image-level, n=498)",
                 fontsize=12.5, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)

    # Annotate spread
    ai_range = df[score_col].max() - df[score_col].min()
    human_range = df[human_col].max() - df[human_col].min()
    ax.text(0.98, 0.97,
            f"AI score range: {df[score_col].min():.2f}–{df[score_col].max():.2f} (span {ai_range:.2f})\n"
            f"Human rating range: {df[human_col].min():.2f}–{df[human_col].max():.2f} (span {human_range:.2f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {fname}")


# =================================================================
# Fig 5: Per-feature correlation bar chart (horizontal, diverging color)
# =================================================================
def make_feature_correlation_bar(csv_path, fname):
    df = pd.read_csv(csv_path)
    # expects columns: Feature, r, p
    df = df.sort_values("r")

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=200)
    colors = [BLUE if v >= 0 else RED for v in df["r"]]
    y_pos = np.arange(len(df))

    ax.barh(y_pos, df["r"], color=colors, zorder=3, height=0.65)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Feature"].str.replace("-Score", ""), fontsize=9.5)
    style_axes(ax, hide_top_right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Pearson r vs Human Rating")
    ax.set_title("Per-Feature Correlation with Human Rating\n(20 geometric features, Combined module)",
                 fontsize=12.5, pad=12)

    # mark significance (p < 0.05) with a small dot
    for i, (r, p) in enumerate(zip(df["r"], df["p"])):
        if p < 0.05:
            ax.text(r + (0.012 if r >= 0 else -0.012), i, "*",
                    va="center", ha="left" if r >= 0 else "right",
                    fontsize=13, color=INK_PRIMARY, fontweight="bold")

    ax.text(0.99, -0.07, "* p < 0.05", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=INK_MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {fname}")


# =================================================================
# Run all
# =================================================================
if __name__ == "__main__":
    make_scatter(
        "Combined_image_level.csv", "Combined-Geometric-Score-1to10", "Human-Rating-Avg",
        n_expected=498,
        title="AI Combined Score vs Human Rating (Image-level, n=498)",
        fname="fig1_scatter_image_level.png",
        point_size=18,
    )
    make_scatter(
        "Combined_subject_level.csv", "Mean-Combined-Geometric-Score-1to10", "Mean-Human-Rating",
        n_expected=100,
        title="AI Combined Score vs Human Rating (Subject-level, n=100)",
        fname="fig2_scatter_subject_level.png",
        point_size=32,
    )
    make_naive_vs_regression_bar()
    make_distribution_comparison(
        "Combined_image_level.csv", "Combined-Geometric-Score-1to10", "Human-Rating-Avg",
        fname="fig4_distribution_comparison.png",
    )
    make_feature_correlation_bar(
        "feature_correlations.csv",
        fname="fig5_feature_correlation_bar.png",
    )
    print("\nAll figures saved to:", OUT_DIR)
