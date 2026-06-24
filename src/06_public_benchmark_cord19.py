#!/usr/bin/env python3
"""
06_public_benchmark_cord19.py

Purpose
-------
Evaluate the SPECTER-Triage ranking pipeline on an external biomedical
retrieval benchmark: CORD-19/TREC-COVID.

This demonstrates generalisation beyond the single mutational-signature
systematic review dataset.

Dataset
-------
CORD-19/TREC-COVID (via ir_datasets):
  - 193K biomedical article abstracts
  - 50 COVID-19 clinical research queries
  - 69K relevance judgments (3-level: 0=not relevant, 1=partially, 2=relevant)

Evaluation
----------
For each query, all methods produce a ranking over the full document collection.
We evaluate with standard IR metrics: MAP, nDCG@10, nDCG@20, P@100, P@200,
Recall@100, Recall@200.

Usage
-----
Install: pip install ir_datasets
Run:     python src/06_public_benchmark_cord19.py

Outputs:
  outputs/public_benchmark/cord19_ranking_scores.csv
  outputs/public_benchmark/cord19_metrics.csv
  outputs/public_benchmark/cord19_table_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ir_datasets
    HAS_IR_DATASETS = True
except ImportError:
    HAS_IR_DATASETS = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


OUTPUT_DIR = Path("outputs/public_benchmark")
EMBEDDING_DIR = Path("outputs/public_benchmark/embeddings")

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER_MODEL_NAME = "allenai-specter"

RELEVANCE_THRESHOLD = 1


def clean_text(value):
    import re
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(text):
    import re
    text = clean_text(text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return [token for token in text.split() if token]


def load_cord19_trec_covid():
    print("Loading CORD-19/TREC-COVID via ir_datasets...")
    dataset = ir_datasets.load("cord19/trec-covid")

    print("Loading documents...")
    docs = {}
    for doc in dataset.docs_iter():
        if doc.abstract:
            docs[doc.doc_id] = {
                "doc_id": doc.doc_id,
                "title": doc.title or "",
                "abstract": doc.abstract,
                "doi": doc.doi or "",
            }
    print(f"  Loaded {len(docs)} documents with abstracts")

    print("Loading queries...")
    queries = {}
    for q in dataset.queries_iter():
        queries[q.query_id] = {
            "query_id": q.query_id,
            "title": q.title,
            "description": q.description,
            "narrative": q.narrative,
        }
    print(f"  Loaded {len(queries)} queries")

    print("Loading relevance judgments...")
    qrels = []
    for qrel in dataset.qrels_iter():
        qrels.append({
            "query_id": qrel.query_id,
            "doc_id": qrel.doc_id,
            "relevance": qrel.relevance,
        })
    qrels_df = pd.DataFrame(qrels)
    print(f"  Loaded {len(qrels_df)} relevance judgments")

    return docs, queries, qrels_df


def build_query_text(query):
    parts = [query["title"]]
    if query.get("description"):
        parts.append(query["description"])
    return " ".join(parts)


def evaluate_query(y_true, scores, k_values=(10, 20, 50, 100, 200)):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]

    n_relevant = int(np.sum(y_true))
    if n_relevant == 0:
        return {}

    metrics = {}
    metrics["ap"] = float(np.sum(np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1) * y_sorted) / n_relevant)

    ideal_order = np.argsort(y_true)[::-1]
    y_ideal = y_true[ideal_order]

    for k in k_values:
        if k > len(y_true):
            continue
        precision_k = float(np.mean(y_sorted[:k]))
        recall_k = float(np.sum(y_sorted[:k]) / n_relevant)

        dcg = float(np.sum(y_sorted[:k] / np.log2(np.arange(2, k + 2))))
        idcg = float(np.sum(y_ideal[:k] / np.log2(np.arange(2, k + 2))))
        ndcg_k = dcg / idcg if idcg > 0 else 0.0

        metrics[f"precision_at_{k}"] = precision_k
        metrics[f"recall_at_{k}"] = recall_k
        metrics[f"ndcg_at_{k}"] = ndcg_k

    return metrics


def run_bm25_ranking(doc_texts, query_text, doc_ids):
    tokenized_docs = [tokenize(text) for text in doc_texts]
    tokenized_query = tokenize(query_text)
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    return dict(zip(doc_ids, scores))


def run_tfidf_ranking(doc_texts, query_text, doc_ids):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(doc_texts + [query_text])
    doc_matrix = matrix[:-1]
    query_vector = matrix[-1]
    scores = cosine_similarity(doc_matrix, query_vector).ravel()
    return dict(zip(doc_ids, scores))


def run_embedding_ranking(doc_texts, query_text, doc_ids, model_name, use_sentence_transformer=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_sentence_transformer:
        model = SentenceTransformer(model_name)
        doc_embs = model.encode(doc_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        query_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        def mean_pool(output, mask):
            embs = output.last_hidden_state
            mask_exp = mask.unsqueeze(-1).expand(embs.size()).float()
            return torch.sum(embs * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)

        def encode_batch(texts, batch_size=8):
            all_embs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    out = model(**enc)
                    pooled = mean_pool(out, enc["attention_mask"])
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    all_embs.append(pooled.cpu().numpy())
            return np.vstack(all_embs)

        doc_embs = encode_batch(doc_texts)
        query_emb = encode_batch([query_text])

    scores = cosine_similarity(doc_embs, query_emb).ravel()
    return dict(zip(doc_ids, scores))


def main():
    if not HAS_IR_DATASETS:
        raise ImportError("Install ir_datasets: pip install ir_datasets")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    docs, queries, qrels_df = load_cord19_trec_covid()

    doc_ids = list(docs.keys())
    doc_texts = [f"{docs[did]['title']} {docs[did]['abstract']}" for did in doc_ids]

    all_metrics_rows = []
    all_ranking_rows = []

    query_ids = sorted(queries.keys())
    print(f"\nEvaluating {len(query_ids)} queries...\n")

    for qid in query_ids:
        query = queries[qid]
        query_text = build_query_text(query)

        qrel_docs = qrels_df[qrels_df["query_id"] == qid]
        relevant_docs = set(qrel_docs[qrel_docs["relevance"] >= RELEVANCE_THRESHOLD]["doc_id"].tolist())

        y_true = np.array([1 if did in relevant_docs else 0 for did in doc_ids])

        n_rel = int(np.sum(y_true))
        print(f"Query {qid}: '{query['title'][:80]}...' ({n_rel} relevant / {len(doc_ids)} docs)")

        if n_rel == 0:
            print(f"  Skipping (no relevant docs)")
            continue

        scores_dict = {}

        if HAS_BM25:
            scores_dict["bm25"] = run_bm25_ranking(doc_texts, query_text, doc_ids)

        scores_dict["tfidf"] = run_tfidf_ranking(doc_texts, query_text, doc_ids)

        scores_dict["minilm"] = run_embedding_ranking(
            doc_texts, query_text, doc_ids,
            MINILM_MODEL_NAME, use_sentence_transformer=True,
        )

        scores_dict["specter"] = run_embedding_ranking(
            doc_texts, query_text, doc_ids,
            SPECTER_MODEL_NAME, use_sentence_transformer=True,
        )

        for method_name, scores_map in scores_dict.items():
            scores_arr = np.array([scores_map[did] for did in doc_ids])

            metrics = evaluate_query(y_true, scores_arr)
            metrics["query_id"] = qid
            metrics["query_title"] = query["title"]
            metrics["method"] = method_name
            metrics["n_relevant"] = n_rel
            all_metrics_rows.append(metrics)

            order = np.argsort(scores_arr)[::-1]
            for rank_idx, doc_idx in enumerate(order[:200]):
                did = doc_ids[doc_idx]
                all_ranking_rows.append({
                    "query_id": qid,
                    "method": method_name,
                    "rank": rank_idx + 1,
                    "doc_id": did,
                    "title": docs[did]["title"],
                    "score": float(scores_arr[doc_idx]),
                    "is_relevant": int(y_true[doc_idx]),
                })

        print(f"  Methods evaluated: {list(scores_dict.keys())}")

    metrics_df = pd.DataFrame(all_metrics_rows)
    ranking_df = pd.DataFrame(all_ranking_rows)

    ranking_df.to_csv(OUTPUT_DIR / "cord19_ranking_scores.csv", index=False)
    print(f"\nSaved per-query rankings: {OUTPUT_DIR / 'cord19_ranking_scores.csv'}")

    summary_rows = []
    for method in metrics_df["method"].unique():
        method_df = metrics_df[metrics_df["method"] == method]
        row = {"method": method, "n_queries": len(method_df)}

        for col in method_df.columns:
            if col in ("method", "query_id", "query_title", "n_queries"):
                continue
            if method_df[col].dtype in (np.float64, np.int64, float, int):
                row[f"mean_{col}"] = method_df[col].mean()
                row[f"std_{col}"] = method_df[col].std()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "cord19_metrics.csv", index=False)
    print(f"Saved summary metrics: {OUTPUT_DIR / 'cord19_metrics.csv'}")

    display_cols = ["method", "n_queries", "mean_ap", "mean_ndcg_at_10", "mean_ndcg_at_20",
                     "mean_precision_at_100", "mean_recall_at_100"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print("\n=== CORD-19/TREC-COVID Summary (mean across 50 queries) ===")
    print(summary_df[display_cols].to_string(index=False))

    summary_df.to_csv(OUTPUT_DIR / "cord19_table_metrics.csv", index=False)
    print(f"\nSaved table: {OUTPUT_DIR / 'cord19_table_metrics.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
