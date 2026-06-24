#!/usr/bin/env python3
"""
10_generate_extension_comparison_table.py

Purpose
-------
Generate the extension comparison table for the paper.
Side-by-side comparison of all methods on both the main dataset
and the CLEF-TAR external benchmark.

Outputs:
  outputs/tables/table_extension_comparison_main.csv
  outputs/tables/table_extension_comparison_clef_tar.csv
  outputs/tables/table_extension_comparison_combined.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


OUTPUT_DIR = Path("outputs/tables")


def compute_rank(scores):
    return np.argsort(scores)[::-1] + 1


def recovery_depth(y_true, scores, target):
    total = int(np.sum(y_true))
    if total == 0:
        return None
    required = int(np.ceil(total * target))
    order = np.argsort(scores)[::-1]
    cumrel = np.cumsum(y_true[order])
    hits = np.where(cumrel >= required)[0]
    return int(hits[0] + 1) if len(hits) > 0 else len(y_true)


def ndcg_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    ideal_order = np.argsort(y_true)[::-1]
    y_ideal = y_true[ideal_order]
    dcg = float(np.sum(y_sorted[:k] / np.log2(np.arange(2, k + 2))))
    idcg = float(np.sum(y_ideal[:k] / np.log2(np.arange(2, k + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Main Dataset ───────────────────────────────────────────────
    print("=== Main Dataset Comparison ===")
    main_df = pd.read_csv("outputs/ranking_scores_with_learned_reranker.csv")
    y_main = main_df["is_relevant"].to_numpy(dtype=int)

    main_methods = {
        "BM25": "bm25_score",
        "TF-IDF": "tfidf_score",
        "MiniLM": "minilm_score",
        "PubMedBERT": "pubmedbert_score",
        "SPECTER": "specter_score",
        "MedCPT": "medcpt_score",
        "SPECTER-hybrid": "specter_hybrid_score",
        "TAR TF-IDF+LogReg": "tar_tfidf_logreg_score",
        "Learned Logistic": "learned_logistic_specter_triage_oof_score",
        "Learned NB": "learned_nb_specter_triage_oof_score",
        "Learned RF": "learned_rf_specter_triage_oof_score",
        "TAR-Augmented ExtraTrees": "learned_extratrees_specter_triage_oof_score",
    }

    main_rows = []
    for name, col in main_methods.items():
        if col not in main_df.columns:
            continue
        scores = main_df[col].fillna(0).to_numpy()
        ap = average_precision_score(y_main, scores)
        r50 = recovery_depth(y_main, scores, 0.50)
        r75 = recovery_depth(y_main, scores, 0.75)
        r90 = recovery_depth(y_main, scores, 0.90)

        order = np.argsort(scores)[::-1]
        rel100 = int(np.sum(y_main[order[:100]]))
        p100 = rel100 / 100
        r100 = rel100 / max(int(np.sum(y_main)), 1)
        n100 = ndcg_at_k(y_main, scores, 100)

        main_rows.append({
            "method": name,
            "AP": round(ap, 4),
            "P@100": round(p100, 3),
            "R@100": round(r100, 3),
            "nDCG@100": round(n100, 4),
            "Rel@100": rel100,
            "Rank@50%": r50,
            "Rank@75%": r75,
            "Rank@90%": r90,
        })

    main_table = pd.DataFrame(main_rows).sort_values("AP", ascending=False)
    main_table.to_csv(OUTPUT_DIR / "table_extension_comparison_main.csv", index=False)
    print(main_table.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'table_extension_comparison_main.csv'}")

    # ── CLEF-TAR External Benchmark ────────────────────────────────
    print("\n\n=== CLEF-TAR External Benchmark Comparison ===")
    clef_df = pd.read_csv("outputs/public_benchmark/clef_tar_ranking_scores_with_oof.csv")
    y_clef = clef_df["is_relevant"].to_numpy(dtype=int)

    clef_methods = {
        "BM25": "bm25_score",
        "TF-IDF": "tfidf_score",
        "MiniLM": "minilm_score",
        "SPECTER": "specter_score",
        "MedCPT": "medcpt_score",
        "TAR TF-IDF+LogReg": "tar_tfidf_logreg_oof_score",
        "Learned ExtraTrees": "learned_extratrees_oof_score",
        "TAR-Augmented ExtraTrees": "tar_augmented_extratrees_oof_score",
    }

    clef_rows = []
    for name, col in clef_methods.items():
        if col not in clef_df.columns:
            continue
        topic_rows = []
        for _, topic_df in clef_df.groupby("topic_id"):
            y_topic = topic_df["is_relevant"].to_numpy(dtype=int)
            scores = topic_df[col].fillna(0).to_numpy()
            order = np.argsort(scores)[::-1]
            rel100 = int(np.sum(y_topic[order[:100]]))
            topic_rows.append({
                "AP": average_precision_score(y_topic, scores),
                "P@100": rel100 / 100,
                "R@100": rel100 / max(int(np.sum(y_topic)), 1),
                "nDCG@100": ndcg_at_k(y_topic, scores, 100),
                "Rel@100": rel100,
                "Rank@50%": recovery_depth(y_topic, scores, 0.50),
                "Rank@75%": recovery_depth(y_topic, scores, 0.75),
                "Rank@90%": recovery_depth(y_topic, scores, 0.90),
            })
        topic_table = pd.DataFrame(topic_rows)

        clef_rows.append({
            "method": name,
            "AP": round(topic_table["AP"].mean(), 4),
            "Std AP": round(topic_table["AP"].std(ddof=1), 4),
            "P@100": round(topic_table["P@100"].mean(), 3),
            "R@100": round(topic_table["R@100"].mean(), 3),
            "nDCG@100": round(topic_table["nDCG@100"].mean(), 4),
            "Rel@100": round(topic_table["Rel@100"].mean(), 2),
            "Rank@50%": round(topic_table["Rank@50%"].mean(), 2),
            "Rank@75%": round(topic_table["Rank@75%"].mean(), 2),
            "Rank@90%": round(topic_table["Rank@90%"].mean(), 2),
        })

    clef_table = pd.DataFrame(clef_rows).sort_values("AP", ascending=False)
    clef_table.to_csv(OUTPUT_DIR / "table_extension_comparison_clef_tar.csv", index=False)
    print(clef_table.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'table_extension_comparison_clef_tar.csv'}")

    # ── Combined Table ─────────────────────────────────────────────
    print("\n\n=== Combined Comparison ===")
    main_pivot = main_table.set_index("method")[["AP", "Rel@100", "Rank@90%"]].copy()
    main_pivot.columns = ["AP_main", "Rel@100_main", "Rank@90_main"]

    clef_pivot = clef_table.set_index("method")[["AP", "Rel@100", "Rank@90%"]].copy()
    clef_pivot.columns = ["AP_clef", "Rel@100_clef", "Rank@90_clef"]

    combined = main_pivot.join(clef_pivot, how="outer").reset_index()
    combined = combined.sort_values("AP_main", ascending=False, na_position="last")
    combined.to_csv(OUTPUT_DIR / "table_extension_comparison_combined.csv", index=False)
    print(combined.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'table_extension_comparison_combined.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
