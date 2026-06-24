#!/usr/bin/env python3
"""
09b_generate_extension_figures.py

Purpose
-------
Generate publication-ready screening burden and recall curve figures for the
extension paper, including the TAR-style TF-IDF Logistic Regression baseline.

Outputs:
  outputs/figures/recall_curve_all_methods.png/pdf
  outputs/figures/screening_burden_comparison.png/pdf
  outputs/figures/active_learning_vs_static_recall.png/pdf
  outputs/figures/screening_fraction_comparison.png/pdf
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# Use the merged file that includes TAR and learned reranker scores.
INPUT_PATH = Path("outputs/ranking_scores_with_learned_reranker.csv")
AL_PATH = Path("outputs/active_learning_simulation.csv")
CLEF_TAR_AL_PATH = Path("outputs/public_benchmark/clef_tar_active_learning.csv")

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("outputs/figures")


# Main-paper figure should be readable, not overcrowded.
# Important methods are darker/thicker; secondary baselines are softer.
METHOD_ORDER = [
    # {
    #     "col": "bm25_score",
    #     "label": "BM25",
    #     "color": "#9AA0A6",
    #     "ls": "--",
    #     "lw": 1.4,
    #     "alpha": 0.65,
    #     "zorder": 1,
    # },
    # {
    #     "col": "tfidf_score",
    #     "label": "TF-IDF",
    #     "color": "#B0BEC5",
    #     "ls": "--",
    #     "lw": 1.4,
    #     "alpha": 0.60,
    #     "zorder": 1,
    # },
    # {
    #     "col": "minilm_score",
    #     "label": "MiniLM",
    #     "color": "#5E97F6",
    #     "ls": "-",
    #     "lw": 1.8,
    #     "alpha": 0.80,
    #     "zorder": 2,
    # },
    {
        "col": "specter_score",
        "label": "SPECTER",
        "color": "#26A69A",
        "ls": "-",
        "lw": 1.8,
        "alpha": 0.80,
        "zorder": 2,
    },
    {
        "col": "specter_hybrid_score",
        "label": "Manual SPECTER-hybrid",
        "color": "#F9A825",
        "ls": "-",
        "lw": 2.4,
        "alpha": 0.95,
        "zorder": 4,
    },
    {
        "col": "tar_tfidf_logreg_score",
        "label": "TAR TF-IDF + LogReg",
        "color": "#6D4C41",
        "ls": "-.",
        "lw": 2.7,
        "alpha": 0.98,
        "zorder": 5,
    },
    {
        "col": "learned_extratrees_specter_triage_oof_score",
        "label": "TAR-Augmented ExtraTrees",
        "color": "#D32F2F",
        "ls": "-",
        "lw": 1.8,
        "alpha": 1.00,
        "zorder": 6,
    },
]


def find_label_col(df):
    for col in ["is_relevant", "relevant", "included", "label"]:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found. Available columns: {df.columns.tolist()}")


def compute_recall_curve(y_true, scores):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    cumrel = np.cumsum(y_sorted)
    total_rel = int(np.sum(y_true))
    recall = cumrel / total_rel if total_rel > 0 else cumrel
    return np.arange(1, len(recall) + 1), recall


def compute_recovery_depth(y_true, scores, target):
    total = int(np.sum(y_true))
    if total == 0:
        return None

    required = int(np.ceil(total * target))
    order = np.argsort(scores)[::-1]
    cumrel = np.cumsum(y_true[order])
    hits = np.where(cumrel >= required)[0]

    if len(hits) == 0:
        return len(y_true)

    return int(hits[0] + 1)


def style_axes(ax):
    """Clean academic figure styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    ax.tick_params(axis="both", labelsize=10, colors="#333333")
    ax.grid(True, axis="y", color="#DADCE0", linewidth=0.8, alpha=0.75)
    ax.grid(True, axis="x", color="#E8EAED", linewidth=0.5, alpha=0.35)


def add_recall_targets(ax):
    """Add subtle horizontal target-recall guide lines."""
    targets = [
        (0.50, "50%"),
        (0.75, "75%"),
        (0.90, "90%"),
    ]

    for y, label in targets:
        ax.axhline(
            y=y,
            color="#777777",
            linestyle=":",
            linewidth=1.0,
            alpha=0.55,
            zorder=0,
        )
        ax.text(
            8,
            y + 0.012,
            label,
            fontsize=9,
            color="#666666",
            va="bottom",
            ha="left",
        )


def generate_recall_curve(df, y_true, available_methods, n):
    print("\nGenerating polished cumulative recall curve...")

    fig, ax = plt.subplots(figsize=(8.8, 5.3))

    max_records = min(500, n)

    recovery_90 = {}

    for method in available_methods:
        col = method["col"]
        scores = df[col].fillna(0).to_numpy()

        x, recall = compute_recall_curve(y_true, scores)
        x_plot = x[:max_records]
        y_plot = recall[:max_records]

        ax.plot(
            x_plot,
            y_plot,
            label=method["label"],
            color=method["color"],
            linestyle=method["ls"],
            linewidth=method["lw"],
            alpha=method["alpha"],
            zorder=method["zorder"],
            drawstyle="steps-post",
        )

        recovery_90[method["label"]] = compute_recovery_depth(y_true, scores, 0.90)

    add_recall_targets(ax)

    # Highlight the key paper claim: ExtraTrees reaches 90% recall earlier than TAR LogReg.
    highlight_methods = {
        "TAR-Augmented ExtraTrees": "#D32F2F",
        "TAR TF-IDF + LogReg": "#6D4C41",
        "Manual SPECTER-hybrid": "#F9A825",
    }

    order = 0.1

    for label, color in highlight_methods.items():
        depth = recovery_90.get(label)
        if depth is None or depth > max_records:
            continue

        ax.axvline(
            x=depth,
            ymin=0,
            ymax=0.90 / 1.05,
            color=color,
            linestyle=":",
            linewidth=1.25,
            alpha=0.65,
            zorder=0,
        )

        ax.scatter(
            [depth],
            [0.90],
            s=52,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=10,
        )

        ax.text(
            depth + 6,
            0.90 - order,
            f"{label}\nRank@90% = {depth}",
            fontsize=8.5,
            color=color,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.90,
            },
        )
        order += 0.09

    ax.set_title(
        "Cumulative Recall vs Records Screened",
        fontsize=14,
        fontweight="bold",
        color="#202124",
        pad=12,
    )

    ax.set_xlabel("Records screened", fontsize=11, color="#202124")
    ax.set_ylabel("Cumulative recall", fontsize=11, color="#202124")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.set_xlim(0, max_records)
    ax.set_ylim(0, 1.03)

    style_axes(ax)

    legend = ax.legend(
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.8,
        labelspacing=0.45,
        handlelength=2.4,
    )
    legend.get_frame().set_edgecolor("#DADCE0")
    legend.get_frame().set_linewidth(0.8)

    fig.text(
        0.01,
        0.01,
        "Note: step curves show cumulative relevant-study recovery under each ranking method.",
        fontsize=8.2,
        color="#5F6368",
    )

    plt.tight_layout(rect=[0, 0.035, 1, 1])

    png_path = FIGURE_DIR / "recall_curve_all_methods.png"
    pdf_path = FIGURE_DIR / "recall_curve_all_methods.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def generate_screening_burden_comparison(y_true, available_methods, n):
    print("\nGenerating screening burden comparison...")

    methods_labels = []
    r50_vals, r75_vals, r90_vals = [], [], []

    for method in available_methods:
        scores = method["scores"]
        methods_labels.append(method["label"])
        r50_vals.append(compute_recovery_depth(y_true, scores, 0.50) or n)
        r75_vals.append(compute_recovery_depth(y_true, scores, 0.75) or n)
        r90_vals.append(compute_recovery_depth(y_true, scores, 0.90) or n)

    x = np.arange(len(methods_labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    ax.bar(x - width, r50_vals, width, label="50% recall", color="#64B5F6")
    ax.bar(x, r75_vals, width, label="75% recall", color="#FFB74D")
    ax.bar(x + width, r90_vals, width, label="90% recall", color="#E57373")

    ax.set_ylabel("Records screened", fontsize=11)
    ax.set_title("Screening Burden to Reach Target Recall", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9, frameon=True)

    style_axes(ax)

    plt.tight_layout()

    png_path = FIGURE_DIR / "screening_burden_comparison.png"
    pdf_path = FIGURE_DIR / "screening_burden_comparison.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    return r50_vals, r75_vals, r90_vals


def generate_active_learning_curve(df, y_true, available_methods, n, total_rel):
    if not AL_PATH.exists():
        return

    print("\nGenerating active learning recall curve...")

    al_df = pd.read_csv(AL_PATH)

    fig, ax = plt.subplots(figsize=(8.8, 5.3))

    batch_colors = {
        25: "#1976D2",
        50: "#388E3C",
        100: "#7B1FA2",
    }

    for bs in sorted(al_df["batch_size"].unique()):
        bs_df = al_df[al_df["batch_size"] == bs]
        color = batch_colors.get(int(bs), "#424242")

        ax.plot(
            bs_df["total_screened"],
            bs_df["recall"],
            label=f"Active learning, batch={bs}",
            linewidth=2.4,
            color=color,
            alpha=0.95,
            drawstyle="steps-post",
        )

    # Keep only the static baselines central to the paper claim.
    static_cols = {
        "specter_hybrid_score": ("Manual SPECTER-hybrid static", "#F9A825"),
        "tar_tfidf_logreg_score": ("TAR TF-IDF + LogReg static", "#6D4C41"),
        "learned_extratrees_specter_triage_oof_score": ("TAR-Augmented ExtraTrees", "#D32F2F"),
    }

    for col, (label, color) in static_cols.items():
        if col not in df.columns:
            continue

        x, recall = compute_recall_curve(y_true, df[col].fillna(0).to_numpy())
        ax.plot(
            x[:500],
            recall[:500],
            label=label,
            color=color,
            linestyle=":",
            linewidth=1.8,
            alpha=0.85,
            drawstyle="steps-post",
        )

    # Random screening reference
    rng = np.random.RandomState(42)
    random_curve = np.cumsum(rng.permutation(y_true)) / total_rel
    ax.plot(
        np.arange(1, min(500, n) + 1),
        random_curve[:min(500, n)],
        color="#9AA0A6",
        linestyle="--",
        linewidth=1.3,
        alpha=0.80,
        label="Random screening",
    )

    add_recall_targets(ax)

    ax.set_title(
        "Active Learning vs Static Ranking",
        fontsize=14,
        fontweight="bold",
        color="#202124",
        pad=12,
    )
    ax.set_xlabel("Records screened", fontsize=11)
    ax.set_ylabel("Cumulative recall", fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.03)

    style_axes(ax)

    legend = ax.legend(
        loc="lower right",
        fontsize=8.2,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
    )
    legend.get_frame().set_edgecolor("#DADCE0")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    png_path = FIGURE_DIR / "active_learning_vs_static_recall.png"
    pdf_path = FIGURE_DIR / "active_learning_vs_static_recall.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def generate_screening_fraction_comparison(y_true, available_methods, n):
    print("\nGenerating screening fraction comparison...")

    pct_50, pct_75, pct_90 = [], [], []

    for method in available_methods:
        scores = method["scores"]
        pct_50.append((compute_recovery_depth(y_true, scores, 0.50) or n) / n * 100)
        pct_75.append((compute_recovery_depth(y_true, scores, 0.75) or n) / n * 100)
        pct_90.append((compute_recovery_depth(y_true, scores, 0.90) or n) / n * 100)

    labels = [m["label"] for m in available_methods]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    ax.bar(x - width, pct_50, width, label="50% recall", color="#64B5F6")
    ax.bar(x, pct_75, width, label="75% recall", color="#FFB74D")
    ax.bar(x + width, pct_90, width, label="90% recall", color="#E57373")

    ax.set_ylabel("% of collection screened", fontsize=11)
    ax.set_title("Screening Fraction to Reach Target Recall", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9, frameon=True)

    style_axes(ax)

    plt.tight_layout()

    png_path = FIGURE_DIR / "screening_fraction_comparison.png"
    pdf_path = FIGURE_DIR / "screening_fraction_comparison.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    label_col = find_label_col(df)
    y_true = df[label_col].to_numpy(dtype=int)

    n = len(df)
    total_rel = int(np.sum(y_true))

    print(f"Records: {n}, Relevant: {total_rel}")

    available_methods = []

    for method in METHOD_ORDER:
        col = method["col"]

        if col not in df.columns:
            print(f"Skipping missing column: {col}")
            continue

        method = method.copy()
        method["scores"] = df[col].fillna(0).to_numpy()
        available_methods.append(method)

    print(f"Available methods: {[m['label'] for m in available_methods]}")

    generate_recall_curve(df, y_true, available_methods, n)
    r50_vals, r75_vals, r90_vals = generate_screening_burden_comparison(
        y_true,
        available_methods,
        n,
    )
    generate_active_learning_curve(df, y_true, available_methods, n, total_rel)
    generate_screening_fraction_comparison(y_true, available_methods, n)

    print("\n=== Screening Depth Summary ===")
    print(f"{'Method':<32} {'50% Recall':>12} {'75% Recall':>12} {'90% Recall':>12}")
    print("-" * 74)

    for i, method in enumerate(available_methods):
        print(
            f"{method['label']:<32} "
            f"{r50_vals[i]:>12} "
            f"{r75_vals[i]:>12} "
            f"{r90_vals[i]:>12}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
