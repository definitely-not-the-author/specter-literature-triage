#!/usr/bin/env python3
"""
05d_run_medcpt_baseline.py

Purpose
-------
Add MedCPT as a retrieval-tuned biomedical dense retrieval baseline.

MedCPT (NCBI) is specifically trained for biomedical literature retrieval
and provides a stronger comparison than mean-pooled PubMedBERT.

Model: ncbi/MedCPT-Query-Encoder (queries) + ncbi/MedCPT-Article-Encoder (documents)

Usage
-----
python src/05d_run_medcpt_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PATH = Path("data/processed/ranking_dataset.csv")
OUTPUT_DIR = Path("outputs/rankings")
EMBEDDING_DIR = Path("outputs/embeddings")

QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"

RESEARCH_QUERY = """
computational methods for mutational signature analysis in cancer genomics,
including mutation signature extraction, signature assignment, non-negative
matrix factorisation, machine learning, deep learning, graph neural networks,
Bayesian models, benchmarking, genomic mutation patterns, mutational processes,
HIV-associated cancer, and cancer genome analysis
"""


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    summed_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / summed_mask


def encode_texts(texts, tokenizer, model, device, batch_size=8, max_length=512):
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            pooled = mean_pooling(output, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy())

            done = min(start + batch_size, len(texts))
            print(f"  Encoded {done}/{len(texts)}")

    return np.vstack(all_embeddings)


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


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["combined_text"] = df["combined_text"].fillna("").astype(str)

    print(f"Loaded ranking dataset: {len(df)} rows")
    print(df["screening_label"].value_counts())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load MedCPT query encoder
    print(f"\nLoading query encoder: {QUERY_MODEL_NAME}")
    query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
    query_model = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(device)

    # Load MedCPT article encoder
    print(f"Loading article encoder: {ARTICLE_MODEL_NAME}")
    article_tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL_NAME)
    article_model = AutoModel.from_pretrained(ARTICLE_MODEL_NAME).to(device)

    # Encode query
    print("\nEncoding research query...")
    query_embedding = encode_texts(
        [RESEARCH_QUERY],
        tokenizer=query_tokenizer,
        model=query_model,
        device=device,
        batch_size=1,
    )

    # Encode documents (with caching)
    article_embedding_path = EMBEDDING_DIR / "medcpt_article_embeddings.npy"

    if article_embedding_path.exists():
        print(f"\nLoading cached MedCPT article embeddings: {article_embedding_path}")
        doc_embeddings = np.load(article_embedding_path)
    else:
        print(f"\nEncoding {len(df)} documents with MedCPT article encoder...")
        doc_embeddings = encode_texts(
            df["combined_text"].tolist(),
            tokenizer=article_tokenizer,
            model=article_model,
            device=device,
            batch_size=8,
        )
        np.save(article_embedding_path, doc_embeddings)
        print(f"Saved MedCPT article embeddings: {article_embedding_path}")

    # Compute cosine similarity
    scores = cosine_similarity(doc_embeddings, query_embedding).ravel()

    df["medcpt_score"] = scores

    save_ranking(df, "medcpt_score", "medcpt")

    print(f"\nMedCPT score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"MedCPT mean: {scores.mean():.4f}, std: {scores.std():.4f}")

    print("\nTop 10 MedCPT papers:")
    top = df.sort_values("medcpt_score", ascending=False).head(10)
    print(top[["record_id", "screening_label", "is_relevant", "title", "medcpt_score"]].to_string(index=False))

    print("\nDone. Added MedCPT ranking.")


if __name__ == "__main__":
    main()
