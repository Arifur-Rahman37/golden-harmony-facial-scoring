"""
make_paper_charts.py
=====================
Generates publication-quality figures for the Golden Harmony paper:
  1. golden_naive_vs_regression.png   - Golden Ratio: naive vs regression scatter (2 panels)
  2. combined_naive_vs_regression.png - Combined: naive vs regression scatter (2 panels)
  3. correlation_summary_bar.png      - Bar chart comparing r across modules/methods
  4. score_distributions.png          - Histograms: AI score vs Human rating spread (why AI range looks narrow)

Run from the same folder as the CSV outputs. Saves all PNGs at 300 DPI.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

POINT_COLOR = "#2E6F9E"   # muted blue
LINE_COLOR = "#D1495B"    # muted red/accent for trend line
POINT_ALPHA = 0.55


def scatter_panel(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=18, color=POINT_COLOR, alpha=POINT_ALPHA, edgecolors="none")
    # trend line
    if len(x) > 2:
        slope, intercept, r, p, se = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color=LINE_COLOR, linewidth=2)
        ax.text(
            0.04, 0.96, f"r = {r:.3f}\np = {p:.2g}\nn = {len(x)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        )
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def make_naive_vs_regression(csv_path, naive_col, reg_col, human_col, title_prefix, out_path):
    df = pd.read_csv(csv_path)
    sub = df.dropna(subset=[naive_col, reg_col, human_col])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    scatter_panel(
        axes[0], sub[naive_col], sub[human_col],
        f"{title_prefix}: Naive (equal-weight)", "AI score (naive, 1-10)", "Human rating (1-10)",
    )
    scatter_panel(
        axes[1], sub[reg_col], sub[human_col],
        f"{title_prefix}: Regression-weighted", "AI score (regression, 1-10)", "Human rating (1-10)",
    )
    fig.suptitle(f"{title_prefix}: AI score vs. Human rating", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_correlation_summary_bar(out_path):
    rows = [
        ("Golden Ratio", "Naive", 0.082),
        ("Golden Ratio", "Regression (CV)", 0.322),
        ("Proportion+\nSymmetry", "Naive", 0.055),
        ("Proportion+\nSymmetry", "Regression (CV)", -0.066),
        ("Combined", "Naive", 0.091),
        ("Combined", "Regression (CV)", 0.295),
    ]
    df = pd.DataFrame(rows, columns=["Module", "Method", "r"])

    modules = df["Module"].unique().tolist()
    methods = ["Naive", "Regression (CV)"]
    colors = {"Naive": "#B0B7C1", "Regression (CV)": "#2E6F9E"}  # gray vs blue

    x = np.arange(len(modules))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        vals = [df[(df.Module == m) & (df.Method == method)]["r"].values[0] for m in modules]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=method, color=colors[method])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + (0.015 if v >= 0 else -0.035),
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9,
            )

    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.set_ylabel("Pearson r (vs. calibrated human rating)")
    ax.set_title("Naive averaging vs. regression-weighted scoring, by module\n(cross-validated r, image-level)", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_score_distributions(combined_csv, out_path):
    df = pd.read_csv(combined_csv)
    sub = df.dropna(subset=["Naive-Average-Score-1to10", "Combined-Geometric-Score-1to10", "Human-Rating-Avg"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        ("Naive-Average-Score-1to10", "AI score (naive average)", "#B0B7C1"),
        ("Combined-Geometric-Score-1to10", "AI score (regression)", "#2E6F9E"),
        ("Human-Rating-Avg", "Human rating (calibrated)", "#D1495B"),
    ]
    for ax, (col, label, color) in zip(axes, specs):
        ax.hist(sub[col], bins=20, color=color, edgecolor="white")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Score (1-10)")
        ax.set_xlim(0.5, 10.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Number of images")
    fig.suptitle("Score distributions: why the naive AI score looks \"narrow\"", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_naive_vs_regression(
        "Golden_Regression_image_level_v2.csv",
        naive_col="Naive-Golden-Rating-1to10", reg_col="Golden-Rating-1to10", human_col="Human-Rating-Avg",
        title_prefix="Golden Ratio", out_path="golden_naive_vs_regression.png",
    )
    make_naive_vs_regression(
        "Combined_technical.csv",
        naive_col="Naive-Average-Score-1to10", reg_col="Combined-Geometric-Score-1to10", human_col="Human-Rating-Avg",
        title_prefix="Combined (20 features)", out_path="combined_naive_vs_regression.png",
    )
    make_correlation_summary_bar("correlation_summary_bar.png")
    make_score_distributions("Combined_technical.csv", "score_distributions.png")
