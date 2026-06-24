from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RANKINGS_DIR = Path("outputs/rankings")
SCORES_PATH = Path("outputs/ranking_scores_with_learned_reranker.csv")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")

SCREENING_EFFICIENCY_TABLE = TABLE_DIR / "table_screening_efficiency.csv"
RECOVERY_DEPTH_TABLE = TABLE_DIR / "table_recovery_depth.csv"

CUMULATIVE_FIGURE = FIGURE_DIR / "cumulative_relevant_retrieval.png"
ENRICHMENT_FIGURE = FIGURE_DIR / "enrichment_at_k.png"


METHOD_FILES = {
    "keyword": RANKINGS_DIR / "ranking_keyword.csv",
    "tfidf": RANKINGS_DIR / "ranking_tfidf.csv",
    "bm25": RANKINGS_DIR / "ranking_bm25.csv",
    "minilm": RANKINGS_DIR / "ranking_minilm.csv",
    "pubmedbert": RANKINGS_DIR / "ranking_pubmedbert.csv",
    "specter": RANKINGS_DIR / "ranking_specter.csv",
    "specter_hybrid": RANKINGS_DIR / "ranking_specter_hybrid.csv",
}

SCORE_METHODS = {
    "tar_tfidf_logreg": "tar_tfidf_logreg_score",
    "tar_augmented_extratrees": "learned_extratrees_specter_triage_oof_score",
}


METHOD_LABELS = {
    "keyword": "Keyword",
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "minilm": "MiniLM",
    "pubmedbert": "PubMedBERT",
    "specter": "SPECTER",
    "specter_hybrid": "SPECTER-hybrid",
    "tar_tfidf_logreg": "TAR TF-IDF+LogReg",
    "tar_augmented_extratrees": "TAR-Augmented ExtraTrees",
}


K_VALUES = [10, 25, 50, 100, 200]
RECALL_TARGETS = [0.25, 0.50, 0.75, 0.90]


def load_rankings():
    rankings = {}

    for method, path in METHOD_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing ranking file for {method}: {path}")

        df = pd.read_csv(path)
        required_cols = ["rank", "record_id", "is_relevant", "score"]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{method} missing required columns: {missing}")

        df = df.sort_values("rank").reset_index(drop=True)
        df["is_relevant"] = df["is_relevant"].astype(int)
        rankings[method] = df

    if SCORES_PATH.exists():
        score_df = pd.read_csv(SCORES_PATH)
        for method, col in SCORE_METHODS.items():
            if col not in score_df.columns:
                continue
            ranked = score_df.copy()
            ranked["score"] = ranked[col].fillna(0)
            ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            ranked["is_relevant"] = ranked["is_relevant"].astype(int)
            rankings[method] = ranked

    return rankings


def find_rank_for_recall(cumulative_relevant, total_relevant, target_recall):
    target_count = int(np.ceil(total_relevant * target_recall))

    positions = np.where(cumulative_relevant >= target_count)[0]

    if len(positions) == 0:
        return None

    # ranks are 1-indexed
    return int(positions[0] + 1)


def create_screening_efficiency_table(rankings):
    first_method = next(iter(rankings))
    n_records = len(rankings[first_method])
    total_relevant = int(rankings[first_method]["is_relevant"].sum())
    dataset_prevalence = total_relevant / n_records

    rows = []

    for method, df in rankings.items():
        relevance = df["is_relevant"].to_numpy()
        cumulative_relevant = np.cumsum(relevance)

        row = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "n_records": n_records,
            "total_relevant": total_relevant,
            "dataset_prevalence": dataset_prevalence,
        }

        for k in K_VALUES:
            relevant_at_k = int(cumulative_relevant[k - 1])
            precision_at_k = relevant_at_k / k
            recall_at_k = relevant_at_k / total_relevant
            screening_fraction_at_k = k / n_records
            enrichment_at_k = precision_at_k / dataset_prevalence

            row[f"relevant_at_{k}"] = relevant_at_k
            row[f"precision_at_{k}"] = precision_at_k
            row[f"recall_at_{k}"] = recall_at_k
            row[f"screening_fraction_at_{k}"] = screening_fraction_at_k
            row[f"enrichment_at_{k}"] = enrichment_at_k

        rows.append(row)

    table = pd.DataFrame(rows)

    # Helpful compact columns for the paper's main screening-efficiency table.
    compact_cols = [
        "method_label",
        "relevant_at_100",
        "recall_at_100",
        "precision_at_100",
        "screening_fraction_at_100",
        "enrichment_at_100",
        "relevant_at_200",
        "recall_at_200",
        "precision_at_200",
        "enrichment_at_200",
    ]

    compact = table[compact_cols].copy()

    rounding_cols = [
        "recall_at_100",
        "precision_at_100",
        "screening_fraction_at_100",
        "enrichment_at_100",
        "recall_at_200",
        "precision_at_200",
        "enrichment_at_200",
    ]

    for col in rounding_cols:
        compact[col] = compact[col].round(3)

    return table, compact


def create_recovery_depth_table(rankings):
    first_method = next(iter(rankings))
    n_records = len(rankings[first_method])
    total_relevant = int(rankings[first_method]["is_relevant"].sum())

    rows = []

    for method, df in rankings.items():
        relevance = df["is_relevant"].to_numpy()
        cumulative_relevant = np.cumsum(relevance)

        row = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
        }

        for target in RECALL_TARGETS:
            rank_needed = find_rank_for_recall(
                cumulative_relevant=cumulative_relevant,
                total_relevant=total_relevant,
                target_recall=target,
            )

            target_pct = int(target * 100)
            row[f"rank_for_{target_pct}_recall"] = rank_needed

            if rank_needed is None:
                row[f"screening_fraction_for_{target_pct}_recall"] = None
            else:
                row[f"screening_fraction_for_{target_pct}_recall"] = rank_needed / n_records

        rows.append(row)

    table = pd.DataFrame(rows)

    for target in RECALL_TARGETS:
        target_pct = int(target * 100)
        fraction_col = f"screening_fraction_for_{target_pct}_recall"
        if fraction_col in table.columns:
            table[fraction_col] = table[fraction_col].round(3)

    return table


def plot_cumulative_retrieval(rankings):
    plt.figure(figsize=(8, 5))

    for method, df in rankings.items():
        relevance = df["is_relevant"].to_numpy()
        cumulative_relevant = np.cumsum(relevance)

        plt.plot(
            np.arange(1, len(cumulative_relevant) + 1),
            cumulative_relevant,
            label=METHOD_LABELS.get(method, method),
        )

    plt.xlabel("Number of records screened")
    plt.ylabel("Included studies retrieved")
    plt.title("Cumulative retrieval of included studies by ranking method")
    plt.xlim(0, 250)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CUMULATIVE_FIGURE, dpi=300)
    plt.close()

    print(f"Saved: {CUMULATIVE_FIGURE}")


def plot_enrichment_at_k(full_efficiency_table):
    plt.figure(figsize=(8, 5))

    for _, row in full_efficiency_table.iterrows():
        method_label = row["method_label"]
        enrichments = [row[f"enrichment_at_{k}"] for k in K_VALUES]

        plt.plot(K_VALUES, enrichments, marker="o", label=method_label)

    plt.xlabel("Top-k ranked records")
    plt.ylabel("Enrichment over dataset prevalence")
    plt.title("Enrichment of included studies among top-ranked records")
    plt.xticks(K_VALUES)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ENRICHMENT_FIGURE, dpi=300)
    plt.close()

    print(f"Saved: {ENRICHMENT_FIGURE}")


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rankings = load_rankings()

    full_efficiency, compact_efficiency = create_screening_efficiency_table(rankings)
    recovery_depth = create_recovery_depth_table(rankings)

    full_efficiency.to_csv(TABLE_DIR / "table_screening_efficiency_full.csv", index=False)
    compact_efficiency.to_csv(SCREENING_EFFICIENCY_TABLE, index=False)
    recovery_depth.to_csv(RECOVERY_DEPTH_TABLE, index=False)

    print(f"Saved: {TABLE_DIR / 'table_screening_efficiency_full.csv'}")
    print(f"Saved: {SCREENING_EFFICIENCY_TABLE}")
    print(f"Saved: {RECOVERY_DEPTH_TABLE}")

    print("\nScreening efficiency table:")
    print(compact_efficiency.to_string(index=False))

    print("\nRecovery depth table:")
    print(recovery_depth.to_string(index=False))

    plot_cumulative_retrieval(rankings)
    plot_enrichment_at_k(full_efficiency)


if __name__ == "__main__":
    main()
