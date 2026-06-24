from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


INPUT_PATH = Path("data/processed/ranking_dataset.csv")
OUTPUT_DIR = Path("outputs/rankings")


RESEARCH_QUERY = """
computational methods for mutational signature analysis in cancer genomics,
including mutation signature extraction, signature assignment, non-negative
matrix factorisation, machine learning, deep learning, graph neural networks,
Bayesian models, benchmarking, genomic mutation patterns, mutational processes,
HIV-associated cancer, and cancer genome analysis
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


def tokenize(text):
    text = clean_text(text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return [token for token in text.split() if token]


def keyword_score(text):
    text = clean_text(text)
    group_scores = []

    for _, keywords in KEYWORD_GROUPS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        group_scores.append(hits / max(len(keywords), 1))

    # Average coverage across concept groups.
    return float(np.mean(group_scores))


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
        "abstract",
    ]

    out[columns].to_csv(OUTPUT_DIR / f"ranking_{method_name}.csv", index=False)
    print(f"Saved {method_name}: {OUTPUT_DIR / f'ranking_{method_name}.csv'}")


def run_keyword(df):
    df = df.copy()
    df["keyword_score"] = df["combined_text"].map(keyword_score)
    save_ranking(df, "keyword_score", "keyword")


def run_tfidf(df):
    df = df.copy()
    documents = df["combined_text"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    matrix = vectorizer.fit_transform(documents + [RESEARCH_QUERY])
    doc_matrix = matrix[:-1]
    query_vector = matrix[-1]

    scores = cosine_similarity(doc_matrix, query_vector).ravel()
    df["tfidf_score"] = scores

    save_ranking(df, "tfidf_score", "tfidf")


def run_bm25(df):
    if not HAS_BM25:
        print("BM25 skipped: install rank-bm25 first.")
        return

    df = df.copy()
    tokenized_docs = [tokenize(text) for text in df["combined_text"].fillna("")]
    tokenized_query = tokenize(RESEARCH_QUERY)

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    df["bm25_score"] = scores
    save_ranking(df, "bm25_score", "bm25")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "record_id",
        "title",
        "abstract",
        "doi",
        "combined_text",
        "screening_label",
        "is_relevant",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded ranking dataset: {len(df)} rows")
    print(df["screening_label"].value_counts())
    print("\nRunning baselines...")

    run_keyword(df)
    run_tfidf(df)
    run_bm25(df)

    print("\nDone.")


if __name__ == "__main__":
    main()