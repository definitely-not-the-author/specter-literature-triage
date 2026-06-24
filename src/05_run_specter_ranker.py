from pathlib import Path
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PATH = Path("data/processed/ranking_dataset.csv")
OUTPUT_DIR = Path("outputs/rankings")
EMBEDDING_DIR = Path("outputs/embeddings")


MODEL_NAME = "allenai-specter"

RESEARCH_QUERY = """
computational methods for mutational signature analysis in cancer genomics,
including mutation signature extraction, signature assignment, non-negative
matrix factorisation, machine learning, deep learning, graph neural networks,
Bayesian models, benchmarking, genomic mutation patterns, mutational processes,
HIV-associated cancer, and cancer genome analysis
"""

PROPOSAL_SUMMARY = """
This review focuses on computational and bioinformatics methods for analysing
mutational signatures and mutational processes. It prioritises method-oriented
studies covering algorithms, statistical models, machine learning, deep learning,
graph-based modelling, signature extraction, decomposition, assignment,
benchmarking, and evaluation in cancer and HIV-associated cancer genomics.
"""

KEYWORD_GROUPS = {
    "mutational_signature": [
        "mutational signature",
        "mutational signatures",
        "mutation signature",
        "mutation signatures",
        "mutational process",
        "mutational processes",
        "mutation pattern",
        "mutation patterns",
        "somatic mutation",
        "somatic mutations",
    ],
    "computational_methods": [
        "computational",
        "algorithm",
        "algorithms",
        "model",
        "models",
        "framework",
        "pipeline",
        "method",
        "methods",
        "statistical",
        "bayesian",
        "machine learning",
        "deep learning",
        "neural network",
        "graph neural network",
        "gnn",
        "non-negative matrix factorization",
        "non-negative matrix factorisation",
        "nmf",
    ],
    "genomics_cancer": [
        "genome",
        "genomes",
        "genomic",
        "cancer",
        "tumor",
        "tumour",
        "oncology",
        "pan-cancer",
        "whole-genome",
        "whole exome",
        "sequencing",
    ],
    "evaluation": [
        "benchmark",
        "benchmarking",
        "validation",
        "evaluate",
        "evaluation",
        "performance",
        "comparison",
        "accuracy",
        "precision",
        "recall",
    ],
    "hiv_context": [
        "hiv",
        "aids",
        "human immunodeficiency virus",
        "immunodeficiency",
        "viral",
        "virus-associated",
    ],
}


def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def keyword_score(text):
    text = clean_text(text)
    group_scores = []

    for _, keywords in KEYWORD_GROUPS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        group_scores.append(hits / max(len(keywords), 1))

    return float(np.mean(group_scores))


def minmax_scale(values):
    values = np.asarray(values, dtype=float)
    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def save_ranking(df, score_col, method_name):
    out = df.copy()
    out["method"] = method_name
    out["score"] = out[score_col]
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    columns = [
        "rank",
        "method",
        "record_id",
        "title",
        "doi",
        "screening_label",
        "is_relevant",
        "score",
        "specter_rq_similarity",
        "specter_proposal_similarity",
        "keyword_score",
        "abstract",
    ]

    out[columns].to_csv(OUTPUT_DIR / f"ranking_{method_name}.csv", index=False)
    print(f"Saved {method_name}: {OUTPUT_DIR / f'ranking_{method_name}.csv'}")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["combined_text"] = df["combined_text"].fillna("").astype(str)

    print(f"Loaded ranking dataset: {len(df)} rows")
    print(df["screening_label"].value_counts())

    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embedding_path = EMBEDDING_DIR / "specter_document_embeddings.npy"

    if embedding_path.exists():
        print(f"Loading cached embeddings: {embedding_path}")
        document_embeddings = np.load(embedding_path)
    else:
        print("Encoding documents with SPECTER...")
        document_embeddings = model.encode(
            df["combined_text"].tolist(),
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(embedding_path, document_embeddings)
        print(f"Saved embeddings: {embedding_path}")

    print("Encoding anchor texts...")
    anchor_embeddings = model.encode(
        [RESEARCH_QUERY, PROPOSAL_SUMMARY],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    rq_embedding = anchor_embeddings[0].reshape(1, -1)
    proposal_embedding = anchor_embeddings[1].reshape(1, -1)

    df["specter_rq_similarity"] = cosine_similarity(
        document_embeddings, rq_embedding
    ).ravel()

    df["specter_proposal_similarity"] = cosine_similarity(
        document_embeddings, proposal_embedding
    ).ravel()

    df["keyword_score"] = df["combined_text"].map(keyword_score)

    # SPECTER-only score: research question anchor only.
    df["specter_score"] = df["specter_rq_similarity"]

    # Hybrid score: scaled semantic and keyword features.
    rq_scaled = minmax_scale(df["specter_rq_similarity"])
    proposal_scaled = minmax_scale(df["specter_proposal_similarity"])
    keyword_scaled = minmax_scale(df["keyword_score"])

    df["specter_hybrid_score"] = (
        0.65 * rq_scaled
        + 0.10 * proposal_scaled
        + 0.25 * keyword_scaled
    )

    save_ranking(df, "specter_score", "specter")
    save_ranking(df, "specter_hybrid_score", "specter_hybrid")

    print("\nTop 10 SPECTER-hybrid papers:")
    top = df.sort_values("specter_hybrid_score", ascending=False).head(10)
    print(
        top[
            [
                "record_id",
                "screening_label",
                "is_relevant",
                "title",
                "specter_hybrid_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()