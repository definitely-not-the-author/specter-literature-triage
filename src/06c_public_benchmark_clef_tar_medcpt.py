#!/usr/bin/env python3
"""
06c_public_benchmark_clef_tar_medcpt.py

Purpose
-------
Run MedCPT (biomedical retrieval model) on the CLEF eHealth TAR benchmark.

This is separated from 06b to keep individual scripts fast.
MedCPT uses two separate encoders (query + article) and is slower than
sentence-transformer based models.

Prerequisites
-------------
- Run 06b first to cache PubMed abstracts
- MedCPT models are downloaded on first run (~400MB total)

Usage
-----
python src/06c_public_benchmark_clef_tar_medcpt.py

Outputs:
  outputs/public_benchmark/clef_tar_medcpt_metrics.csv
  outputs/public_benchmark/clef_tar_medcpt_table_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ir_datasets
except ImportError:
    pass

from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModel


CLEF_TAR_DIR = Path("data/public_benchmark/clef-tar")
OUTPUT_DIR = Path("outputs/public_benchmark")

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"


def load_clef_tar_topics():
    topics_dir = CLEF_TAR_DIR / "training" / "topics_train"
    topics = {}
    for filepath in sorted(topics_dir.iterdir()):
        if filepath.is_file() and not filepath.name.startswith("."):
            topic = parse_topic_file(filepath)
            if "topic_id" in topic:
                topics[topic["topic_id"]] = topic
    print(f"  Loaded {len(topics)} topics")
    return topics


def parse_topic_file(filepath):
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    topic = {"pids": []}
    current_key = None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Topic:"):
            topic["topic_id"] = stripped.split(":", 1)[1].strip()
            current_key = None
        elif stripped.startswith("Title:"):
            topic["title"] = stripped.split(":", 1)[1].strip()
            current_key = None
        elif stripped.startswith("Query:"):
            current_key = "query"
            topic["query"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Pids:"):
            current_key = "pids"
        elif current_key == "query" and stripped:
            topic["query"] = topic.get("query", "") + " " + stripped
        elif current_key == "pids" and stripped:
            pid = stripped.split()[0] if stripped.split() else ""
            if pid and pid.isdigit():
                topic["pids"].append(pid)
    return topic


def load_clef_tar_qrels():
    qrels_dir = CLEF_TAR_DIR / "training" / "qrels"
    qrels_files = [f for f in qrels_dir.iterdir() if f.is_file() and not f.name.startswith(".")]

    preferred = ["qrel_abs_train", "train.abs.qrels", "train.abs.rels"]
    qrels_path = None
    for name in preferred:
        for f in qrels_files:
            if f.name == name:
                qrels_path = f
                break
        if qrels_path:
            break
    if qrels_path is None:
        qrels_path = sorted(qrels_files)[0]

    rows = []
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                rows.append({
                    "topic_id": parts[0],
                    "iteration": parts[1],
                    "doc_id": parts[2],
                    "relevance": int(parts[3]),
                })
    return pd.DataFrame(rows)


def load_cached_abstracts():
    cache_path = CLEF_TAR_DIR / "pubmed_abstracts.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run 06b_public_benchmark_clef_tar.py first to fetch PubMed abstracts."
        )
    cache_df = pd.read_csv(cache_path)
    docs = {}
    for _, row in cache_df.iterrows():
        docs[str(row["pmid"])] = {
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", "")),
            "pmid": str(row["pmid"]),
        }
    print(f"  Loaded {len(docs)} cached documents")
    return docs


def evaluate_query(y_true, scores, k_values=(10, 20, 50, 100)):
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
        metrics[f"precision_at_{k}"] = float(np.mean(y_sorted[:k]))
        metrics[f"recall_at_{k}"] = float(np.sum(y_sorted[:k]) / n_relevant)
        dcg = float(np.sum(y_sorted[:k] / np.log2(np.arange(2, k + 2))))
        idcg = float(np.sum(y_ideal[:k] / np.log2(np.arange(2, k + 2))))
        metrics[f"ndcg_at_{k}"] = dcg / idcg if idcg > 0 else 0.0
    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading CLEF-TAR data ===")
    topics = load_clef_tar_topics()
    qrels_df = load_clef_tar_qrels()
    docs = load_cached_abstracts()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading MedCPT models on {device}...")
    query_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_MODEL)
    query_model = AutoModel.from_pretrained(MEDCPT_QUERY_MODEL).to(device)
    article_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_ARTICLE_MODEL)
    article_model = AutoModel.from_pretrained(MEDCPT_ARTICLE_MODEL).to(device)
    print("  MedCPT models loaded.")

    def mean_pool(output, mask):
        embs = output.last_hidden_state
        mask_exp = mask.unsqueeze(-1).expand(embs.size()).float()
        return torch.sum(embs * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)

    def encode_batch(tokenizer, model, texts, batch_size=8):
        all_embs = []
        model.eval()
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

    all_metrics_rows = []
    topic_ids = sorted(topics.keys())
    print(f"\n=== Evaluating MedCPT on {len(topic_ids)} topics ===\n")

    for tid in topic_ids:
        topic = topics[tid]
        query_text = (topic.get("title", "") + " " + topic.get("query", "")).strip()

        pids_for_topic = topic.get("pids", [])
        qrels_for_topic = qrels_df[qrels_df["topic_id"] == tid]
        relevant_pids = set(qrels_for_topic[qrels_for_topic["relevance"] == 1]["doc_id"].tolist())

        doc_ids_in_topic = []
        doc_texts_in_topic = []
        for pid in pids_for_topic:
            if pid in docs:
                doc = docs[pid]
                text = f"{doc['title']} {doc['abstract']}".strip()
                if text:
                    doc_ids_in_topic.append(pid)
                    doc_texts_in_topic.append(text)

        if not doc_ids_in_topic:
            continue

        y_true = np.array([1 if pid in relevant_pids else 0 for pid in doc_ids_in_topic])
        n_rel = int(np.sum(y_true))
        print(f"Topic {tid}: {len(doc_ids_in_topic)} docs, {n_rel} relevant")

        if n_rel == 0:
            continue

        query_emb = encode_batch(query_tokenizer, query_model, [query_text], batch_size=1)
        doc_embs = encode_batch(article_tokenizer, article_model, doc_texts_in_topic, batch_size=8)
        scores = cosine_similarity(doc_embs, query_emb).ravel()

        metrics = evaluate_query(y_true, scores)
        metrics["topic_id"] = tid
        metrics["topic_title"] = topic.get("title", "")
        metrics["method"] = "medcpt"
        metrics["n_docs"] = len(doc_ids_in_topic)
        metrics["n_relevant"] = n_rel
        all_metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(all_metrics_rows)

    row = {"method": "medcpt", "n_topics": len(metrics_df)}
    for col in metrics_df.columns:
        if col in ("method", "topic_id", "topic_title", "n_topics", "n_docs", "n_relevant"):
            continue
        if metrics_df[col].dtype in (np.float64, np.int64, float, int):
            row[f"mean_{col}"] = metrics_df[col].mean()
            row[f"std_{col}"] = metrics_df[col].std()

    summary_df = pd.DataFrame([row])
    summary_df.to_csv(OUTPUT_DIR / "clef_tar_medcpt_metrics.csv", index=False)

    display_cols = ["method", "n_topics", "mean_ap", "mean_ndcg_at_10", "mean_ndcg_at_20",
                     "mean_precision_at_100", "mean_recall_at_100"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print("\n=== CLEF-TAR MedCPT Summary ===")
    print(summary_df[display_cols].to_string(index=False))

    summary_df.to_csv(OUTPUT_DIR / "clef_tar_medcpt_table_metrics.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'clef_tar_medcpt_table_metrics.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
