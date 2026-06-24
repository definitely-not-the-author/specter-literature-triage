from pathlib import Path
import numpy as np
import pandas as pd


RANKINGS_DIR = Path("outputs/rankings")
OUTPUT_DIR = Path("outputs/metrics")
OUTPUT_PATH = OUTPUT_DIR / "ranking_metrics.csv"

METHOD_FILES = {
    "keyword": RANKINGS_DIR / "ranking_keyword.csv",
    "tfidf": RANKINGS_DIR / "ranking_tfidf.csv",
    "bm25": RANKINGS_DIR / "ranking_bm25.csv",
    "minilm": RANKINGS_DIR / "ranking_minilm.csv",
    "pubmedbert": RANKINGS_DIR / "ranking_pubmedbert.csv",
    "specter": RANKINGS_DIR / "ranking_specter.csv",
    "specter_hybrid": RANKINGS_DIR / "ranking_specter_hybrid.csv",
}

K_VALUES = [10, 25, 50, 100, 200]


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


def first_relevant_rank(relevance):
    for idx, rel in enumerate(relevance, start=1):
        if rel == 1:
            return idx
    return None


def evaluate_method(method_name, path):
    if not path.exists():
        raise FileNotFoundError(f"Missing ranking file for {method_name}: {path}")

    df = pd.read_csv(path)

    required_cols = ["rank", "record_id", "is_relevant", "score"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{method_name} missing columns: {missing}")

    df = df.sort_values("rank").reset_index(drop=True)
    relevance = df["is_relevant"].astype(int).to_numpy()
    total_relevant = int(relevance.sum())

    row = {
        "method": method_name,
        "n_records": len(df),
        "total_relevant": total_relevant,
        "average_precision": average_precision(relevance, total_relevant),
        "first_relevant_rank": first_relevant_rank(relevance),
    }

    for k in K_VALUES:
        row[f"precision_at_{k}"] = precision_at_k(relevance, k)
        row[f"recall_at_{k}"] = recall_at_k(relevance, k, total_relevant)
        row[f"ndcg_at_{k}"] = ndcg_at_k(relevance, k, total_relevant)
        row[f"relevant_found_at_{k}"] = int(np.asarray(relevance[:k]).sum())

    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for method_name, path in METHOD_FILES.items():
        print(f"Evaluating {method_name}: {path}")
        rows.append(evaluate_method(method_name, path))

    metrics = pd.DataFrame(rows)

    # Put strongest summary columns first.
    ordered_cols = [
        "method",
        "n_records",
        "total_relevant",
        "average_precision",
        "first_relevant_rank",
        "precision_at_10",
        "precision_at_25",
        "precision_at_50",
        "precision_at_100",
        "recall_at_50",
        "recall_at_100",
        "recall_at_200",
        "ndcg_at_50",
        "ndcg_at_100",
        "ndcg_at_200",
        "relevant_found_at_50",
        "relevant_found_at_100",
        "relevant_found_at_200",
    ]

    remaining_cols = [col for col in metrics.columns if col not in ordered_cols]
    metrics = metrics[ordered_cols + remaining_cols]

    metrics.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved metrics: {OUTPUT_PATH}")
    print("\nMain metrics:")
    print(
        metrics[
            [
                "method",
                "average_precision",
                "precision_at_25",
                "precision_at_50",
                "recall_at_100",
                "ndcg_at_100",
                "relevant_found_at_100",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()