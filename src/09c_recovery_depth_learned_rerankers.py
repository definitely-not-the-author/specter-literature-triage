#!/usr/bin/env python3
"""
09c_recovery_depth_learned_rerankers.py

Purpose
-------
Extend the recovery depth table to include all learned rerankers
and the TAR TF-IDF+LogReg baseline, not just the 7 standalone baselines.

Usage
-----
python src/09c_recovery_depth_learned_rerankers.py

Outputs:
  outputs/tables/table_recovery_depth_full.csv
  outputs/figures/recovery_depth_comparison.png
  outputs/figures/recovery_depth_comparison.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCORES_PATH = Path("outputs/ranking_scores_with_learned_reranker.csv")
RECALL_TARGETS = [0.25, 0.50, 0.75, 0.90]

METHOD_CONFIGS = {
    "keyword": {"col": "keyword_score", "label": "Keyword"},
    "tfidf": {"col": "tfidf_score", "label": "TF-IDF"},
    "bm25": {"col": "bm25_score", "label": "BM25"},
    "minilm": {"col": "minilm_score", "label": "MiniLM"},
    "pubmedbert": {"col": "pubmedbert_score", "label": "PubMedBERT"},
    "specter": {"col": "specter_score", "label": "SPECTER"},
    "specter_hybrid": {"col": "specter_hybrid_score", "label": "SPECTER-hybrid"},
    "tar_tfidf_logreg": {"col": "tar_tfidf_logreg_score", "label": "TAR TF-IDF+LogReg"},
    "learned_logistic": {"col": "learned_logistic_specter_triage_oof_score", "label": "Learned Logistic"},
    "learned_nb": {"col": "learned_nb_specter_triage_oof_score", "label": "Learned NB"},
    "learned_rf": {"col": "learned_rf_specter_triage_oof_score", "label": "Learned RF"},
    "learned_extratrees": {"col": "learned_extratrees_specter_triage_oof_score", "label": "TAR-Augmented ExtraTrees"},
    "learned_gb": {"col": "learned_gb_specter_triage_oof_score", "label": "Learned GB"},
    "learned_hgb": {"col": "learned_hgb_specter_triage_oof_score", "label": "Learned HGB"},
    "learned_adaboost": {"col": "learned_adaboost_specter_triage_oof_score", "label": "Learned AdaBoost"},
    "learned_svm_linear": {"col": "learned_svm_linear_specter_triage_oof_score", "label": "Learned SVM-Linear"},
}

HIGHLIGHT_METHODS = {"learned_extratrees", "tar_tfidf_logreg", "specter_hybrid"}


def compute_recovery_depth(y_true: np.ndarray, scores: np.ndarray,
                           target_recall: float) -> int:
    total = int(np.sum(y_true))
    if total == 0:
        return len(y_true)
    required = int(np.ceil(total * target_recall))
    order = np.argsort(scores)[::-1]
    cumrel = np.cumsum(y_true[order])
    hits = np.where(cumrel >= required)[0]
    if len(hits) == 0:
        return len(y_true)
    return int(hits[0] + 1)


def main():
    parser = argparse.ArgumentParser(description="Recovery depth with learned rerankers")
    parser.add_argument("--input", default=str(SCORES_PATH))
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)
    y = df["is_relevant"].values.astype(int)
    n = len(df)
    total_rel = int(np.sum(y))
    print(f"Records: {n}, Relevant: {total_rel}")

    rows = []
    for method, cfg in METHOD_CONFIGS.items():
        col = cfg["col"]
        if col not in df.columns:
            print(f"  Skipping {method}: column '{col}' not found")
            continue

        scores = df[col].fillna(0).values
        order = np.argsort(scores)[::-1]
        y_sorted = y[order]

        row = {"method": method, "method_label": cfg["label"]}
        for target in RECALL_TARGETS:
            rank = compute_recovery_depth(y, scores, target)
            frac = rank / n
            row[f"rank_for_{int(target*100)}_recall"] = rank
            row[f"screening_fraction_for_{int(target*100)}_recall"] = round(frac, 4)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(tables_dir / "table_recovery_depth_full.csv", index=False)
    print(f"\nSaved: {tables_dir / 'table_recovery_depth_full.csv'}")
    print(result_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    methods_for_plot = []
    rank_90_vals = []
    colors = []
    palette = {"learned_extratrees": "#0072B2", "tar_tfidf_logreg": "#D55E00",
               "specter_hybrid": "#009E73"}

    for _, row in result_df.iterrows():
        methods_for_plot.append(row["method_label"])
        rank_90_vals.append(row["rank_for_90_recall"])
        colors.append(palette.get(row["method"], "#999999"))

    sorted_idx = np.argsort(rank_90_vals)
    methods_for_plot = [methods_for_plot[i] for i in sorted_idx]
    rank_90_vals = [rank_90_vals[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    ax.barh(methods_for_plot, rank_90_vals, color=colors)
    ax.set_xlabel("Records to screen for 90% recall")
    ax.set_title("Recovery Depth at 90% Recall: All Methods")
    ax.axvline(x=total_rel * 2, color="red", linestyle="--", alpha=0.5, label="2x relevant count")
    fig.tight_layout()
    fig.savefig(figures_dir / "recovery_depth_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / "recovery_depth_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {figures_dir / 'recovery_depth_comparison.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
