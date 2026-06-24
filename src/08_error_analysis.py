#!/usr/bin/env python3
"""
08_error_analysis.py

Purpose
-------
Compare the TAR-Augmented ExtraTrees / SPECTER-Triage reranker against:
  1. Manual SPECTER-hybrid baseline
  2. TAR-style TF-IDF + Logistic Regression baseline, if available

Identifies:
  - Records recovered by ExtraTrees but missed by comparator
  - Records recovered by comparator but missed by ExtraTrees
  - False positives in the ExtraTrees top 100
  - False negatives beyond dynamically computed Rank@90% recall

Usage
-----
python src/08_error_analysis.py

Outputs
-------
outputs/error_analysis_summary.csv
outputs/error_analysis_gained.csv
outputs/error_analysis_lost.csv
outputs/error_analysis_false_positives_top100.csv
outputs/error_analysis_false_negatives.csv
outputs/error_analysis_et_vs_tar_gained.csv
outputs/error_analysis_et_vs_tar_lost.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


INPUT_PATH = Path("outputs/ranking_scores_with_learned_reranker.csv")
OUTPUT_DIR = Path("outputs")

LEARNED_COL = "learned_extratrees_specter_triage_oof_score"
LEARNED_LABEL = "TAR-Augmented ExtraTrees"
HYBRID_COL = "specter_hybrid_score"
TAR_COL = "tar_tfidf_logreg_score"

LABEL_COL = "is_relevant"
TOP_K = 100
TARGET_RECALL = 0.90


def compute_rank(scores: np.ndarray) -> np.ndarray:
    """
    Return per-record ranks.

    rank[i] = rank position of record i, where 1 is the highest score.

    Important:
    np.argsort(scores)[::-1] gives sorted record indices, NOT per-record ranks.
    This function correctly maps sorted order back to each original record.
    """
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def top_k_indices(scores: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Return original record indices for the top-k records."""
    return np.argsort(scores, kind="mergesort")[::-1][:k]


def relevant_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = TOP_K) -> int:
    """Count relevant records in the top-k."""
    idx = top_k_indices(scores, k)
    return int(np.sum(y_true[idx]))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = TOP_K) -> float:
    """Compute recall at k."""
    total_rel = int(np.sum(y_true))
    if total_rel == 0:
        return 0.0
    return relevant_at_k(y_true, scores, k) / total_rel


def rank_at_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> int | None:
    """
    Return the screening rank required to reach target recall.

    Example:
    If this returns 268 for target_recall=0.90, then 90% recall was first
    reached after screening 268 records.
    """
    total_rel = int(np.sum(y_true))
    if total_rel == 0:
        return None

    order = np.argsort(scores, kind="mergesort")[::-1]
    y_sorted = y_true[order]

    cumulative_rel = np.cumsum(y_sorted)
    recall = cumulative_rel / total_rel

    hits = np.where(recall >= target_recall)[0]
    if len(hits) == 0:
        return None

    return int(hits[0] + 1)


def method_metrics(name: str, y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Compute core ranking metrics used in this error analysis."""
    return {
        "method": name,
        "average_precision": average_precision_score(y_true, scores),
        "relevant_at_100": relevant_at_k(y_true, scores, TOP_K),
        "recall_at_100": recall_at_k(y_true, scores, TOP_K),
        "rank_at_90_recall": rank_at_recall(y_true, scores, TARGET_RECALL),
    }


def safe_cols(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    """Keep only columns that exist in a dataframe."""
    return [c for c in cols if c in frame.columns]

def save_subset(
    df: pd.DataFrame,
    indices,
    output_path: Path,
    sort_col: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Save a subset dataframe safely.

    This supports both:
      - original positional indices from the full dataframe
      - existing index labels from an already-subset dataframe
    """
    indices = list(indices)

    if len(indices) == 0:
        subset = df.iloc[[]].copy()
    elif all(i in df.index for i in indices):
        subset = df.loc[indices].copy()
    else:
        subset = df.iloc[indices].copy()

    if sort_col in subset.columns:
        subset = subset.sort_values(sort_col, ascending=ascending)

    preferred_cols = [
        "record_id",
        "title",
        "abstract",
        "screening_label",
        LABEL_COL,
        "et_rank",
        "hybrid_rank",
        "tar_rank",
        "et_score",
        "hybrid_score",
        "tar_score",
        "category",
        "gain_type",
    ]

    subset[safe_cols(subset, preferred_cols)].to_csv(output_path, index=False)
    return subset

def compare_topk(
    *,
    df: pd.DataFrame,
    y_true: np.ndarray,
    et_scores: np.ndarray,
    comparator_scores: np.ndarray,
    comparator_name: str,
    comparator_short_name: str,
    output_gain_path: Path | None = None,
    output_lost_path: Path | None = None,
) -> dict:
    """
    Compare ExtraTrees top-k against another method's top-k.

    Gained:
        In ExtraTrees top-k, not in comparator top-k.
    Lost:
        In comparator top-k, not in ExtraTrees top-k.
    """
    et_top = set(top_k_indices(et_scores, TOP_K))
    comp_top = set(top_k_indices(comparator_scores, TOP_K))

    gained = et_top - comp_top
    lost = comp_top - et_top

    gained_relevant = [i for i in gained if y_true[i] == 1]
    gained_irrelevant = [i for i in gained if y_true[i] == 0]
    lost_relevant = [i for i in lost if y_true[i] == 1]
    lost_irrelevant = [i for i in lost if y_true[i] == 0]

    print(f"\n=== ExtraTrees vs {comparator_name} Top-{TOP_K} Comparison ===")
    print(f"  ExtraTrees top {TOP_K}: {len(et_top)} records")
    print(f"  {comparator_name} top {TOP_K}: {len(comp_top)} records")
    print(f"  Overlap: {len(et_top & comp_top)} records")
    print(
        f"  Gained by ExtraTrees: {len(gained)} "
        f"({len(gained_relevant)} relevant + {len(gained_irrelevant)} irrelevant)"
    )
    print(
        f"  Lost by ExtraTrees: {len(lost)} "
        f"({len(lost_relevant)} relevant + {len(lost_irrelevant)} irrelevant)"
    )
    print(f"  Net relevant gain: {len(gained_relevant) - len(lost_relevant)}")

    if output_gain_path is not None:
        gained_df = df.iloc[list(gained)].copy()
        gained_df["category"] = gained_df[LABEL_COL].map(
            {1: "true_positive_gain", 0: "false_positive_gain"}
        )
        gained_df["gain_type"] = f"et_vs_{comparator_short_name}_gained"
        save_subset(gained_df, gained_df.index, output_gain_path, "et_score")
        print(f"  Saved: {output_gain_path}")

    if output_lost_path is not None:
        lost_df = df.iloc[list(lost)].copy()
        lost_df["category"] = lost_df[LABEL_COL].map(
            {1: "true_positive_lost", 0: "false_positive_lost"}
        )
        lost_df["gain_type"] = f"et_vs_{comparator_short_name}_lost"

        sort_col = "tar_score" if comparator_short_name == "tar" else "hybrid_score"
        save_subset(lost_df, lost_df.index, output_lost_path, sort_col)
        print(f"  Saved: {output_lost_path}")

    return {
        "comparison": f"ExtraTrees vs {comparator_name}",
        "comparator": comparator_name,
        "et_top100_relevant": relevant_at_k(y_true, et_scores, TOP_K),
        "comparator_top100_relevant": relevant_at_k(y_true, comparator_scores, TOP_K),
        "overlap_top100": len(et_top & comp_top),
        "gained_total": len(gained),
        "gained_relevant": len(gained_relevant),
        "gained_irrelevant": len(gained_irrelevant),
        "lost_total": len(lost),
        "lost_relevant": len(lost_relevant),
        "lost_irrelevant": len(lost_irrelevant),
        "net_relevant_gain": len(gained_relevant) - len(lost_relevant),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    required_cols = [LABEL_COL, LEARNED_COL, HYBRID_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    y_true = df[LABEL_COL].to_numpy(dtype=int)
    n = len(df)
    total_rel = int(np.sum(y_true))

    print(f"Records: {n}, Relevant: {total_rel}")

    et_scores = df[LEARNED_COL].fillna(0).to_numpy()
    hybrid_scores = df[HYBRID_COL].fillna(0).to_numpy()

    tar_scores = None
    has_tar = TAR_COL in df.columns
    if has_tar:
        tar_scores = df[TAR_COL].fillna(0).to_numpy()

    # Correct per-record ranks
    df["et_rank"] = compute_rank(et_scores)
    df["hybrid_rank"] = compute_rank(hybrid_scores)
    df["et_score"] = et_scores
    df["hybrid_score"] = hybrid_scores

    if has_tar:
        df["tar_rank"] = compute_rank(tar_scores)
        df["tar_score"] = tar_scores

    # === Method metrics ===
    print("\n=== Method Comparison ===")

    metric_rows = [
        method_metrics("manual_specter_hybrid", y_true, hybrid_scores),
        method_metrics("tar_augmented_extratrees", y_true, et_scores),
    ]

    if has_tar:
        metric_rows.append(method_metrics("tar_tfidf_logreg", y_true, tar_scores))

    metrics_df = pd.DataFrame(metric_rows)

    for _, row in metrics_df.iterrows():
        print(
            f"  {row['method']}: "
            f"AP={row['average_precision']:.4f}, "
            f"Relevant@100={int(row['relevant_at_100'])}, "
            f"Recall@100={row['recall_at_100']:.1%}, "
            f"Rank@90%={row['rank_at_90_recall']}"
        )

    metrics_df.to_csv(OUTPUT_DIR / "error_analysis_method_metrics.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'error_analysis_method_metrics.csv'}")

    # === ExtraTrees vs manual SPECTER-hybrid ===
    hybrid_comparison = compare_topk(
        df=df,
        y_true=y_true,
        et_scores=et_scores,
        comparator_scores=hybrid_scores,
        comparator_name="Manual SPECTER-hybrid",
        comparator_short_name="hybrid",
        output_gain_path=OUTPUT_DIR / "error_analysis_gained.csv",
        output_lost_path=OUTPUT_DIR / "error_analysis_lost.csv",
    )

    # === False positives in ExtraTrees top 100 ===
    et_top100_indices = top_k_indices(et_scores, TOP_K)
    fp_top100 = [i for i in et_top100_indices if y_true[i] == 0]

    fp_df = df.iloc[fp_top100].copy()
    fp_df["category"] = "false_positive_top100"
    fp_df["gain_type"] = "et_top100_false_positive"
    fp_df = fp_df.sort_values("et_score", ascending=False)

    fp_output = OUTPUT_DIR / "error_analysis_false_positives_top100.csv"
    save_subset(fp_df, fp_df.index, fp_output, "et_score")
    print(f"\nSaved: {fp_output}")

    # === False negatives beyond dynamically computed ExtraTrees Rank@90% ===
    et_rank90 = rank_at_recall(y_true, et_scores, TARGET_RECALL)

    if et_rank90 is None:
        fn_indices = np.array([], dtype=int)
    else:
        fn_indices = np.where((y_true == 1) & (df["et_rank"].to_numpy() > et_rank90))[0]

    fn_df = df.iloc[fn_indices].copy()
    fn_df["category"] = f"false_negative_beyond_rank{et_rank90}"
    fn_df["gain_type"] = "et_false_negative_after_rank90"
    fn_df = fn_df.sort_values("et_rank", ascending=True)

    fn_output = OUTPUT_DIR / "error_analysis_false_negatives.csv"
    save_subset(fn_df, fn_df.index, fn_output, "et_rank", ascending=True)
    print(f"Saved: {fn_output}")

    # === ExtraTrees vs TAR TF-IDF + Logistic Regression ===
    tar_comparison = None

    if has_tar:
        tar_comparison = compare_topk(
            df=df,
            y_true=y_true,
            et_scores=et_scores,
            comparator_scores=tar_scores,
            comparator_name="TAR TF-IDF + LogReg",
            comparator_short_name="tar",
            output_gain_path=OUTPUT_DIR / "error_analysis_et_vs_tar_gained.csv",
            output_lost_path=OUTPUT_DIR / "error_analysis_et_vs_tar_lost.csv",
        )

    # === Summary CSV ===
    summary_rows = [
        {
            "category": "total_records",
            "count": n,
            "pct_of_total_relevant": np.nan,
        },
        {
            "category": "total_relevant",
            "count": total_rel,
            "pct_of_total_relevant": 1.0,
        },
        {
            "category": "et_top100_relevant",
            "count": relevant_at_k(y_true, et_scores, TOP_K),
            "pct_of_total_relevant": recall_at_k(y_true, et_scores, TOP_K),
        },
        {
            "category": "hybrid_top100_relevant",
            "count": relevant_at_k(y_true, hybrid_scores, TOP_K),
            "pct_of_total_relevant": recall_at_k(y_true, hybrid_scores, TOP_K),
        },
        {
            "category": "et_rank_at_90_recall",
            "count": et_rank90,
            "pct_of_total_relevant": TARGET_RECALL,
        },
        {
            "category": f"fn_beyond_rank_at_{int(TARGET_RECALL * 100)}_recall",
            "count": len(fn_indices),
            "pct_of_total_relevant": len(fn_indices) / total_rel if total_rel else np.nan,
        },
        {
            "category": "fp_top100",
            "count": len(fp_top100),
            "pct_of_total_relevant": np.nan,
        },
        {
            "category": "et_vs_hybrid_gained_relevant",
            "count": hybrid_comparison["gained_relevant"],
            "pct_of_total_relevant": hybrid_comparison["gained_relevant"] / total_rel,
        },
        {
            "category": "et_vs_hybrid_lost_relevant",
            "count": hybrid_comparison["lost_relevant"],
            "pct_of_total_relevant": hybrid_comparison["lost_relevant"] / total_rel,
        },
        {
            "category": "et_vs_hybrid_net_relevant_gain",
            "count": hybrid_comparison["net_relevant_gain"],
            "pct_of_total_relevant": hybrid_comparison["net_relevant_gain"] / total_rel,
        },
        {
            "category": "et_hybrid_top100_overlap",
            "count": hybrid_comparison["overlap_top100"],
            "pct_of_total_relevant": np.nan,
        },
    ]

    if has_tar and tar_comparison is not None:
        summary_rows.extend(
            [
                {
                    "category": "tar_top100_relevant",
                    "count": relevant_at_k(y_true, tar_scores, TOP_K),
                    "pct_of_total_relevant": recall_at_k(y_true, tar_scores, TOP_K),
                },
                {
                    "category": "et_vs_tar_gained_relevant",
                    "count": tar_comparison["gained_relevant"],
                    "pct_of_total_relevant": tar_comparison["gained_relevant"] / total_rel,
                },
                {
                    "category": "et_vs_tar_lost_relevant",
                    "count": tar_comparison["lost_relevant"],
                    "pct_of_total_relevant": tar_comparison["lost_relevant"] / total_rel,
                },
                {
                    "category": "et_vs_tar_net_relevant_gain",
                    "count": tar_comparison["net_relevant_gain"],
                    "pct_of_total_relevant": tar_comparison["net_relevant_gain"] / total_rel,
                },
                {
                    "category": "et_tar_top100_overlap",
                    "count": tar_comparison["overlap_top100"],
                    "pct_of_total_relevant": np.nan,
                },
            ]
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_output = OUTPUT_DIR / "error_analysis_summary.csv"
    summary_df.to_csv(summary_output, index=False)
    print(f"\nSaved: {summary_output}")

    # === Print detailed summary ===
    print("\n=== Error Analysis Summary ===")
    print(f"  Total records: {n}")
    print(f"  Total relevant: {total_rel}")

    et_rel100 = relevant_at_k(y_true, et_scores, TOP_K)
    hybrid_rel100 = relevant_at_k(y_true, hybrid_scores, TOP_K)

    print(f"  {LEARNED_LABEL} relevant@100: {et_rel100} ({et_rel100 / total_rel:.1%})")
    print(f"  Hybrid relevant@100: {hybrid_rel100} ({hybrid_rel100 / total_rel:.1%})")

    if has_tar:
        tar_rel100 = relevant_at_k(y_true, tar_scores, TOP_K)
        print(f"  TAR TF-IDF+LogReg relevant@100: {tar_rel100} ({tar_rel100 / total_rel:.1%})")

    print(
        f"  ET vs Hybrid gained: {hybrid_comparison['gained_relevant']} relevant + "
        f"{hybrid_comparison['gained_irrelevant']} irrelevant"
    )
    print(
        f"  ET vs Hybrid lost: {hybrid_comparison['lost_relevant']} relevant + "
        f"{hybrid_comparison['lost_irrelevant']} irrelevant"
    )
    print(f"  ET vs Hybrid net relevant gain: {hybrid_comparison['net_relevant_gain']}")

    if has_tar and tar_comparison is not None:
        print(
            f"  ET vs TAR gained: {tar_comparison['gained_relevant']} relevant + "
            f"{tar_comparison['gained_irrelevant']} irrelevant"
        )
        print(
            f"  ET vs TAR lost: {tar_comparison['lost_relevant']} relevant + "
            f"{tar_comparison['lost_irrelevant']} irrelevant"
        )
        print(f"  ET vs TAR net relevant gain: {tar_comparison['net_relevant_gain']}")

    print(f"  False positives in ET top 100: {len(fp_top100)}")

    if et_rank90 is not None:
        print(f"  {LEARNED_LABEL} Rank@90% recall: {et_rank90}")
        print(f"  False negatives beyond Rank@90% ({et_rank90}): {len(fn_indices)}")
    else:
        print(f"  {LEARNED_LABEL} Rank@90% recall: not reached")

    # === Heuristic category analysis for false positives ===
    if len(fp_top100) > 0:
        print("\n=== False Positive Categories in ExtraTrees Top 100 ===")

        fp_abstracts = df.iloc[fp_top100]["abstract"].fillna("").astype(str).str.lower()

        cancer_genomics_broad = fp_abstracts.str.contains(
            "cancer|tumor|tumour|oncolog",
            regex=True,
        ).sum()

        mentions_genomics = fp_abstracts.str.contains(
            "genome|genomic|sequencing|mutation",
            regex=True,
        ).sum()

        mentions_signature = fp_abstracts.str.contains(
            "mutational signature|mutation signature",
            regex=True,
        ).sum()

        non_signature_genomic = max(0, int(mentions_genomics - mentions_signature))

        categories = {
            "cancer_genomics_broad": int(cancer_genomics_broad),
            "review_article": int(
                fp_abstracts.str.contains(
                    "review|systematic review|meta-analysis",
                    regex=True,
                ).sum()
            ),
            "non_signature_genomic": non_signature_genomic,
            "clinical_trial": int(
                fp_abstracts.str.contains(
                    "clinical trial|randomized|randomised|placebo",
                    regex=True,
                ).sum()
            ),
            "short_abstract": int(
                (df.iloc[fp_top100]["abstract"].fillna("").astype(str).str.len() < 200).sum()
            ),
        }

        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}/{len(fp_top100)} ({count / len(fp_top100):.0%})")

    # === Suggested manual annotation categories ===
    print("\n=== Error Categories for Manual Annotation ===")
    print("  1. semantic_only_recovery    — dense similarity found it, lexical missed it")
    print("  2. keyword_heavy_recovery    — lexical found it, dense missed it")
    print("  3. off_topic_cancer_genomics — broad cancer genomics, not mutational signatures")
    print("  4. methodologically_relevant — methodology paper but weak domain match")
    print("  5. terminology_mismatch      — relevant topic but different terminology")
    print("  6. short_abstract            — abstract < 200 chars")
    print("  7. broad_review_article      — general review, not focused on methods")
    print("  8. non_signature_genomic     — genomic method, not mutational signatures")

    # === What distinguishes ET gained vs ET lost relevant records? ===
    gained_path = OUTPUT_DIR / "error_analysis_gained.csv"
    lost_path = OUTPUT_DIR / "error_analysis_lost.csv"

    if gained_path.exists() and lost_path.exists():
        gained_file = pd.read_csv(gained_path)
        lost_file = pd.read_csv(lost_path)

        if not gained_file.empty and not lost_file.empty:
            gained_relevant_ids = set(
                gained_file.loc[gained_file[LABEL_COL] == 1, "record_id"]
            ) if "record_id" in gained_file.columns else set()

            lost_relevant_ids = set(
                lost_file.loc[lost_file[LABEL_COL] == 1, "record_id"]
            ) if "record_id" in lost_file.columns else set()

            if gained_relevant_ids and lost_relevant_ids and "record_id" in df.columns:
                gained_df_rel = df[df["record_id"].isin(gained_relevant_ids)]
                lost_df_rel = df[df["record_id"].isin(lost_relevant_ids)]

                print("\n=== ET vs Hybrid: Gained vs Lost Relevant Records ===")

                feature_cols = [
                    "keyword_score",
                    "bm25_score",
                    "minilm_score",
                    "specter_score",
                    "specter_rq_similarity",
                    "specter_proposal_similarity",
                    "medcpt_score",
                    "tar_tfidf_logreg_score",
                ]

                for col in feature_cols:
                    if col in df.columns:
                        g_mean = gained_df_rel[col].mean()
                        l_mean = lost_df_rel[col].mean()
                        print(f"  {col}: gained={g_mean:.4f}, lost={l_mean:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
