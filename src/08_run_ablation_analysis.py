from pathlib import Path
import numpy as np
import pandas as pd


SPECTER_RANKING_PATH = Path("outputs/rankings/ranking_specter_hybrid.csv")
OUTPUT_METRICS = Path("outputs/metrics/ablation_metrics.csv")
OUTPUT_TABLE = Path("outputs/tables/table_ablation_metrics.csv")
OUTPUT_RANKINGS = Path("outputs/rankings/ablation_rankings.csv")


K_VALUES = [10, 25, 50, 100, 200]


VARIANTS = {
    "rq_only": {
        "specter_rq_similarity": 1.0,
        "specter_proposal_similarity": 0.0,
        "keyword_score": 0.0,
    },
    "proposal_only": {
        "specter_rq_similarity": 0.0,
        "specter_proposal_similarity": 1.0,
        "keyword_score": 0.0,
    },
    "keyword_only": {
        "specter_rq_similarity": 0.0,
        "specter_proposal_similarity": 0.0,
        "keyword_score": 1.0,
    },
    "rq_proposal": {
        "specter_rq_similarity": 0.80,
        "specter_proposal_similarity": 0.20,
        "keyword_score": 0.0,
    },
    "rq_keyword": {
        "specter_rq_similarity": 0.75,
        "specter_proposal_similarity": 0.0,
        "keyword_score": 0.25,
    },
    "full_hybrid_original": {
        "specter_rq_similarity": 0.65,
        "specter_proposal_similarity": 0.10,
        "keyword_score": 0.25,
    },
    "semantic_heavy": {
        "specter_rq_similarity": 0.75,
        "specter_proposal_similarity": 0.15,
        "keyword_score": 0.10,
    },
    "balanced": {
        "specter_rq_similarity": 0.50,
        "specter_proposal_similarity": 0.20,
        "keyword_score": 0.30,
    },
    "keyword_heavy": {
        "specter_rq_similarity": 0.45,
        "specter_proposal_similarity": 0.10,
        "keyword_score": 0.45,
    },
}


def minmax_scale(values):
    values = np.asarray(values, dtype=float)
    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def precision_at_k(relevance, k):
    relevance = np.asarray(relevance[:k])
    if len(relevance) == 0:
        return 0.0
    return float(relevance.sum() / len(relevance))


def recall_at_k(relevance, k, total_relevant):
    if total_relevant == 0:
        return 0.0
    relevance = np.asarray(relevance[:k])
    return float(relevance.sum() / total_relevant)


def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance[:k], dtype=float)
    if len(relevance) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance / discounts))


def ndcg_at_k(relevance, k, total_relevant):
    dcg = dcg_at_k(relevance, k)
    ideal_relevance = np.ones(min(k, total_relevant))
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0:
        return 0.0

    return float(dcg / ideal_dcg)


def average_precision(relevance, total_relevant):
    if total_relevant == 0:
        return 0.0

    relevance = np.asarray(relevance)
    precisions = []

    for idx, rel in enumerate(relevance, start=1):
        if rel == 1:
            precisions.append(relevance[:idx].sum() / idx)

    if not precisions:
        return 0.0

    return float(np.sum(precisions) / total_relevant)


def evaluate_ranked_df(df, variant_name, weights):
    relevance = df["is_relevant"].astype(int).to_numpy()
    total_relevant = int(relevance.sum())

    row = {
        "variant": variant_name,
        "weight_rq": weights["specter_rq_similarity"],
        "weight_proposal": weights["specter_proposal_similarity"],
        "weight_keyword": weights["keyword_score"],
        "n_records": len(df),
        "total_relevant": total_relevant,
        "average_precision": average_precision(relevance, total_relevant),
    }

    for k in K_VALUES:
        row[f"precision_at_{k}"] = precision_at_k(relevance, k)
        row[f"recall_at_{k}"] = recall_at_k(relevance, k, total_relevant)
        row[f"ndcg_at_{k}"] = ndcg_at_k(relevance, k, total_relevant)
        row[f"relevant_found_at_{k}"] = int(np.asarray(relevance[:k]).sum())

    return row


def main():
    if not SPECTER_RANKING_PATH.exists():
        raise FileNotFoundError(f"Missing input: {SPECTER_RANKING_PATH}")

    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RANKINGS.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPECTER_RANKING_PATH)

    required_cols = [
        "record_id",
        "title",
        "doi",
        "screening_label",
        "is_relevant",
        "specter_rq_similarity",
        "specter_proposal_similarity",
        "keyword_score",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    scaled = {
        "specter_rq_similarity": minmax_scale(df["specter_rq_similarity"]),
        "specter_proposal_similarity": minmax_scale(df["specter_proposal_similarity"]),
        "keyword_score": minmax_scale(df["keyword_score"]),
    }

    all_metrics = []
    all_rankings = []

    for variant_name, weights in VARIANTS.items():
        working = df.copy()

        score = np.zeros(len(working))

        for feature, weight in weights.items():
            score += weight * scaled[feature]

        working["variant"] = variant_name
        working["ablation_score"] = score
        working = working.sort_values("ablation_score", ascending=False).reset_index(drop=True)
        working["ablation_rank"] = np.arange(1, len(working) + 1)

        all_metrics.append(evaluate_ranked_df(working, variant_name, weights))

        all_rankings.append(
            working[
                [
                    "variant",
                    "ablation_rank",
                    "record_id",
                    "title",
                    "doi",
                    "screening_label",
                    "is_relevant",
                    "ablation_score",
                    "specter_rq_similarity",
                    "specter_proposal_similarity",
                    "keyword_score",
                ]
            ]
        )

    metrics = pd.DataFrame(all_metrics)
    rankings = pd.concat(all_rankings, ignore_index=True)

    metrics = metrics.sort_values(
        ["average_precision", "ndcg_at_100", "precision_at_100"],
        ascending=False,
    )

    table = metrics[
        [
            "variant",
            "weight_rq",
            "weight_proposal",
            "weight_keyword",
            "average_precision",
            "precision_at_25",
            "precision_at_50",
            "precision_at_100",
            "recall_at_100",
            "ndcg_at_100",
            "relevant_found_at_100",
        ]
    ].copy()

    numeric_cols = [
        "average_precision",
        "precision_at_25",
        "precision_at_50",
        "precision_at_100",
        "recall_at_100",
        "ndcg_at_100",
    ]

    for col in numeric_cols:
        table[col] = table[col].round(3)

    metrics.to_csv(OUTPUT_METRICS, index=False)
    table.to_csv(OUTPUT_TABLE, index=False)
    rankings.to_csv(OUTPUT_RANKINGS, index=False)

    print(f"Saved metrics: {OUTPUT_METRICS}")
    print(f"Saved table: {OUTPUT_TABLE}")
    print(f"Saved rankings: {OUTPUT_RANKINGS}")

    print("\nAblation summary:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()