#!/usr/bin/env python3
"""
08b_error_analysis_clef_tar.py

Purpose
-------
Error analysis on the CLEF-TAR external benchmark.

Compares the TAR-Augmented ExtraTrees reranker against MiniLM (best standalone)
to understand when and why the model succeeds/fails on external data.

Automatically categorises errors into:
  - semantic-only recovery (dense similarity found it, lexical missed it)
  - keyword-heavy recovery (lexical found it, dense missed it)
  - high-confidence false positive (model scores non-relevant highly)
  - low-recall relevant (relevant but ranked very low)
  - short abstract (abstract < 200 chars)

Usage
-----
python src/08b_error_analysis_clef_tar.py

Outputs:
  outputs/public_benchmark/clef_tar_error_analysis_summary.csv
  outputs/public_benchmark/clef_tar_error_analysis_gained.csv
  outputs/public_benchmark/clef_tar_error_analysis_lost.csv
  outputs/public_benchmark/clef_tar_error_analysis_fp_top50.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("outputs/public_benchmark/clef_tar_ranking_scores_with_oof.csv")
OUTPUT_DIR = Path("outputs/public_benchmark")

ET_COL = "tar_augmented_extratrees_oof_score"
ET_LABEL = "TAR-Augmented ExtraTrees"
STANDALONE_COL = "minilm_score"
CATEGORIES = [
    "semantic_only_recovery",
    "keyword_heavy_recovery",
    "high_confidence_fp",
    "low_recall_relevant",
    "short_abstract",
]


def categorise_record(row, et_rank, st_rank, is_fp=False, is_fn=False):
    """Assign error category based on feature patterns."""
    categories = []

    if is_fp:
        bm25 = row.get("bm25_score", 0)
        specter = row.get("specter_score", 0)
        minilm = row.get("minilm_score", 0)
        abstract_len = len(str(row.get("abstract", "")))

        if specter > 0.3 or minilm > 0.3:
            if bm25 < 0.1:
                categories.append("semantic_only_recovery")
        if bm25 > 0.1 and minilm < 0.2:
            categories.append("keyword_heavy_recovery")
        categories.append("high_confidence_fp")
        if abstract_len < 200:
            categories.append("short_abstract")

    if is_fn:
        categories.append("low_recall_relevant")
        abstract_len = len(str(row.get("abstract", "")))
        if abstract_len < 200:
            categories.append("short_abstract")

    if not categories:
        if et_rank < st_rank:
            categories.append("semantic_only_recovery")
        else:
            categories.append("keyword_heavy_recovery")

    return "; ".join(categories) if categories else "other"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    y = df["is_relevant"].to_numpy(dtype=int)
    n = len(df)
    n_rel = int(np.sum(y))
    print(f"Records: {n}, Relevant: {n_rel}")

    if ET_COL not in df.columns:
        raise ValueError(
            f"Missing required column: {ET_COL}. "
            "Run src/06d_build_clef_tar_ranking_scores.py before this script."
        )

    et_scores = df[ET_COL].fillna(0).to_numpy()
    st_scores = df[STANDALONE_COL].fillna(0).to_numpy()

    df["et_score"] = et_scores
    df["st_score"] = st_scores

    # Compute ranks per topic
    all_et_ranks = np.zeros(n, dtype=int)
    all_st_ranks = np.zeros(n, dtype=int)

    for tid in df["topic_id"].unique():
        mask = df["topic_id"] == tid
        et_s = et_scores[mask]
        st_s = st_scores[mask]
        all_et_ranks[mask] = np.argsort(np.argsort(-et_s)) + 1
        all_st_ranks[mask] = np.argsort(np.argsort(-st_s)) + 1

    df["et_rank"] = all_et_ranks
    df["st_rank"] = all_st_ranks

    # ── Per-topic analysis ─────────────────────────────────────────
    print("\n=== Per-topic Error Analysis ===")

    all_gained, all_lost, all_fp, all_fn = [], [], [], []

    for tid in df["topic_id"].unique():
        t = df[df["topic_id"] == tid].copy()
        y_t = t["is_relevant"].values
        n_docs = len(t)
        n_rel_t = int(np.sum(y_t))

        if n_rel_t == 0:
            continue

        t_et50 = set(t[t["et_rank"] <= 50].index)
        t_st50 = set(t[t["st_rank"] <= 50].index)
        t_et100 = set(t[t["et_rank"] <= 100].index)
        t_st100 = set(t[t["st_rank"] <= 100].index)

        # Gained: ET finds in top50, standalone misses
        gained_idx = t_et50 - t_st50
        lost_idx = t_st50 - t_et50

        gained_rel = [i for i in gained_idx if y[i] == 1]
        lost_rel = [i for i in lost_idx if y[i] == 1]

        # FP in ET top50
        fp_idx = [i for i in t_et50 if y[i] == 0]
        # FN beyond rank 100
        fn_idx = [i for i in t.index if y[i] == 1 and t.loc[i, "et_rank"] > 100]

        for i in gained_rel:
            row = df.loc[i].to_dict()
            row["category"] = categorise_record(row, row["et_rank"], row["st_rank"])
            row["gain_type"] = "tar_augmented_et_gained"
            all_gained.append(row)

        for i in lost_rel:
            row = df.loc[i].to_dict()
            row["category"] = categorise_record(row, row["et_rank"], row["st_rank"], is_fn=True)
            row["gain_type"] = "tar_augmented_et_lost"
            all_lost.append(row)

        for i in fp_idx:
            row = df.loc[i].to_dict()
            row["category"] = categorise_record(row, row["et_rank"], row["st_rank"], is_fp=True)
            all_fp.append(row)

        for i in fn_idx:
            row = df.loc[i].to_dict()
            row["category"] = categorise_record(row, row["et_rank"], row["st_rank"], is_fn=True)
            all_fn.append(row)

        print(f"  {tid}: {n_docs} docs, {n_rel_t} rel | "
              f"gained={len(gained_rel)}, lost={len(lost_rel)}, "
              f"fp_top50={len(fp_idx)}, fn_beyond100={len(fn_idx)}")

    # ── Save gained/lost/FP/FN ─────────────────────────────────────
    base_cols = ["topic_id", "record_id", "title", "abstract", "is_relevant",
                  "et_rank", "st_rank", "et_score", "st_score",
                  "bm25_score", "minilm_score", "specter_score",
                  "medcpt_score", "category", "gain_type"]

    if all_gained:
        gain_cols = [c for c in base_cols if c in all_gained[0] or c in ("category", "gain_type")]
        pd.DataFrame(all_gained)[gain_cols].to_csv(
            OUTPUT_DIR / "clef_tar_error_analysis_gained.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR / 'clef_tar_error_analysis_gained.csv'} ({len(all_gained)} rows)")

    if all_lost:
        lost_cols = [c for c in base_cols if c in all_lost[0] or c in ("category", "gain_type")]
        pd.DataFrame(all_lost)[lost_cols].to_csv(
            OUTPUT_DIR / "clef_tar_error_analysis_lost.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'clef_tar_error_analysis_lost.csv'} ({len(all_lost)} rows)")

    if all_fp:
        fp_cols = [c for c in base_cols if c in all_fp[0] or c in ("category",)]
        pd.DataFrame(all_fp)[fp_cols].to_csv(
            OUTPUT_DIR / "clef_tar_error_analysis_fp_top50.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'clef_tar_error_analysis_fp_top50.csv'} ({len(all_fp)} rows)")

    if all_fn:
        fn_cols = [c for c in base_cols if c in all_fn[0] or c in ("category",)]
        pd.DataFrame(all_fn)[fn_cols].to_csv(
            OUTPUT_DIR / "clef_tar_error_analysis_fn_beyond100.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'clef_tar_error_analysis_fn_beyond100.csv'} ({len(all_fn)} rows)")

    # ── Summary statistics ──────────────────────────────────────────
    print("\n=== Error Category Distribution ===")

    if all_gained:
        cats = []
        for r in all_gained:
            cats.extend(r["category"].split("; "))
        cat_counts = pd.Series(cats).value_counts()
        print("\nGained records (ET finds, standalone misses):")
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count}")

    if all_lost:
        cats = []
        for r in all_lost:
            cats.extend(r["category"].split("; "))
        cat_counts = pd.Series(cats).value_counts()
        print("\nLost records (standalone finds, ET misses):")
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count}")

    if all_fp:
        cats = []
        for r in all_fp:
            cats.extend(r["category"].split("; "))
        cat_counts = pd.Series(cats).value_counts()
        print("\nFalse positives in ET top 50:")
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count}")

    # ── Summary table ──────────────────────────────────────────────
    summary_rows = [
        {"metric": "total_topics", "value": df["topic_id"].nunique()},
        {"metric": "total_records", "value": n},
        {"metric": "total_relevant", "value": n_rel},
        {"metric": "gained_relevant", "value": len([r for r in all_gained if r["is_relevant"] == 1])},
        {"metric": "lost_relevant", "value": len([r for r in all_lost if r["is_relevant"] == 1])},
        {"metric": "fp_top50", "value": len(all_fp)},
        {"metric": "fn_beyond100", "value": len(all_fn)},
        {"metric": "net_gain_relevant", "value": len([r for r in all_gained if r["is_relevant"] == 1]) - len([r for r in all_lost if r["is_relevant"] == 1])},
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "clef_tar_error_analysis_summary.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'clef_tar_error_analysis_summary.csv'}")

    print("\n=== Summary ===")
    for row in summary_rows:
        print(f"  {row['metric']}: {row['value']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
