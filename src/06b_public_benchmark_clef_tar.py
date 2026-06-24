#!/usr/bin/env python3
"""
06b_public_benchmark_clef_tar.py

Purpose
-------
Evaluate the SPECTER-Triage ranking pipeline on the CLEF eHealth TAR benchmark,
the gold-standard dataset for systematic review screening in empirical medicine.

Dataset
-------
CLEF eHealth TAR (training set, 2017):
  - Cochrane systematic review topics
  - PubMed records with binary include/exclude labels
  - ~149K judged documents, ~2.8K relevant

Prerequisites
-------------
1. Clone the CLEF-TAR repository:
   git clone https://github.com/CLEF-TAR/tar.git data/public_benchmark/clef-tar

2. Install dependencies (if not already):
   pip install ir_datasets  (for CORD-19 script)
   pip install biopython    (optional, for PubMed fetch)

Usage
-----
python src/06b_public_benchmark_clef_tar.py

Outputs:
  outputs/public_benchmark/clef_tar_metrics.csv
  outputs/public_benchmark/clef_tar_table_metrics.csv
  outputs/public_benchmark/clef_tar_ranking_scores.csv
"""

from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

import torch
from sentence_transformers import SentenceTransformer


CLEF_TAR_DIR = Path("data/public_benchmark/clef-tar")
OUTPUT_DIR = Path("outputs/public_benchmark")

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER_MODEL_NAME = "allenai-specter"

PUBMED_BATCH_SIZE = 200
PUBMED_DELAY = 0.35


def clone_clef_tar():
    if CLEF_TAR_DIR.exists():
        print(f"CLEF-TAR repo already exists: {CLEF_TAR_DIR}")
        return

    print("Cloning CLEF-TAR repository...")
    CLEF_TAR_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/CLEF-TAR/tar.git", str(CLEF_TAR_DIR)],
        check=True,
    )
    print(f"Cloned to: {CLEF_TAR_DIR}")


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
            query_text = stripped.split(":", 1)[1].strip()
            topic["query"] = query_text
        elif stripped.startswith("Pids:"):
            current_key = "pids"
        elif current_key == "query" and stripped:
            topic["query"] = topic.get("query", "") + " " + stripped
        elif current_key == "pids" and stripped:
            pid = stripped.split()[0] if stripped.split() else ""
            if pid and pid.isdigit():
                topic["pids"].append(pid)

    return topic


def load_clef_tar_topics():
    topics_dir = CLEF_TAR_DIR / "training" / "topics_train"
    if not topics_dir.exists():
        raise FileNotFoundError(f"Topics directory not found: {topics_dir}")

    topics = {}
    for filepath in sorted(topics_dir.iterdir()):
        if filepath.is_file() and not filepath.name.startswith("."):
            topic = parse_topic_file(filepath)
            if "topic_id" in topic:
                tid = topic["topic_id"]
                topics[tid] = topic
                print(f"  Topic {tid}: {topic.get('title', '')[:80]}")
    print(f"  Total topics: {len(topics)}")
    return topics


def load_clef_tar_qrels():
    qrels_dir = CLEF_TAR_DIR / "training" / "qrels"
    qrels_files = [f for f in qrels_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not qrels_files:
        raise FileNotFoundError(f"No qrels files found in {qrels_dir}")

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

    print(f"Loading qrels from: {qrels_path}")

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

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} relevance judgments")
    print(f"  Topics: {df['topic_id'].nunique()}")
    print(f"  Relevant (rel=1): {(df['relevance'] == 1).sum()}")
    print(f"  Not relevant (rel=0): {(df['relevance'] == 0).sum()}")
    return df


def load_pids_per_topic():
    extracted_dir = CLEF_TAR_DIR / "training" / "extracted_data"
    if not extracted_dir.exists():
        return None

    topic_pids = {}
    for pid_file in extracted_dir.glob("*.pids"):
        topic_id = pid_file.stem
        pids = []
        with open(pid_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    pids.append(parts[-1])
        topic_pids[topic_id] = pids

    print(f"  Loaded PIDs for {len(topic_pids)} topics")
    return topic_pids


def fetch_pubmed_abstracts(pids, batch_size=PUBMED_BATCH_SIZE, delay=PUBMED_DELAY):
    import time
    import urllib.request
    import urllib.parse

    docs = {}
    unique_pids = list(set(pids))

    print(f"  Fetching {len(unique_pids)} unique PubMed abstracts...")

    for i in range(0, len(unique_pids), batch_size):
        batch = unique_pids[i:i+batch_size]
        id_str = ",".join(batch)

        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={id_str}&rettype=abstract&retmode=xml"
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SPECTER-Triage/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_text = resp.read().decode("utf-8", errors="replace")

            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                if pmid_el is None:
                    continue
                pmid = pmid_el.text

                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None and title_el.text else ""

                abstract_parts = []
                for abs_text in article.findall(".//AbstractText"):
                    label = abs_text.get("Label", "")
                    text = abs_text.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                docs[pmid] = {"title": title, "abstract": abstract, "pmid": pmid}

        except Exception as e:
            print(f"  Warning: batch {i//batch_size + 1} failed: {e}")

        done = min(i + batch_size, len(unique_pids))
        print(f"  Fetched {done}/{len(unique_pids)}")

        if i + batch_size < len(unique_pids):
            time.sleep(delay)

    print(f"  Total documents fetched: {len(docs)}")
    return docs


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


def run_bm25_ranking(doc_texts, query_text, doc_ids):
    tokenized_docs = [tokenize(text) for text in doc_texts]
    tokenized_query = tokenize(query_text)
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    return dict(zip(doc_ids, scores))


def run_tfidf_ranking(doc_texts, query_text, doc_ids):
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), min_df=2,
    )
    matrix = vectorizer.fit_transform(doc_texts + [query_text])
    scores = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    return dict(zip(doc_ids, scores))


def run_embedding_ranking(doc_texts, query_text, doc_ids, model_name):
    model = SentenceTransformer(model_name)
    doc_embs = model.encode(doc_texts, batch_size=32, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
    query_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    scores = cosine_similarity(doc_embs, query_emb).ravel()
    return dict(zip(doc_ids, scores))


def main():
    clone_clef_tar()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading CLEF-TAR data ===")
    topics = load_clef_tar_topics()
    qrels_df = load_clef_tar_qrels()

    all_pids = []
    for tid, topic in topics.items():
        all_pids.extend(topic.get("pids", []))
    unique_pids = list(set(all_pids))
    print(f"\nTotal unique PIDs across all topics: {len(unique_pids)}")

    print("\n=== Fetching PubMed abstracts ===")
    abstracts_cache = CLEF_TAR_DIR / "pubmed_abstracts.csv"

    if abstracts_cache.exists():
        print(f"Loading cached abstracts from: {abstracts_cache}")
        cache_df = pd.read_csv(abstracts_cache)
        docs = {}
        for _, row in cache_df.iterrows():
            docs[str(row["pmid"])] = {
                "title": str(row.get("title", "")),
                "abstract": str(row.get("abstract", "")),
                "pmid": str(row["pmid"]),
            }
        print(f"  Loaded {len(docs)} cached documents")

        missing_pids = [p for p in unique_pids if p not in docs]
        if missing_pids:
            print(f"  Fetching {len(missing_pids)} missing abstracts...")
            new_docs = fetch_pubmed_abstracts(missing_pids)
            docs.update(new_docs)
            cache_df = pd.DataFrame([
                {"pmid": v["pmid"], "title": v["title"], "abstract": v["abstract"]}
                for v in docs.values()
            ])
            cache_df.to_csv(abstracts_cache, index=False)
            print(f"  Updated cache: {abstracts_cache}")
        else:
            print("  All abstracts already cached.")
    else:
        docs = fetch_pubmed_abstracts(unique_pids)
        cache_df = pd.DataFrame([
            {"pmid": v["pmid"], "title": v["title"], "abstract": v["abstract"]}
            for v in docs.values()
        ])
        cache_df.to_csv(abstracts_cache, index=False)
        print(f"  Saved {len(docs)} abstracts to: {abstracts_cache}")

    all_metrics_rows = []
    all_ranking_rows = []

    topic_ids = sorted(topics.keys())
    print(f"\n=== Evaluating {len(topic_ids)} topics ===\n")

    for tid in topic_ids:
        topic = topics[tid]
        query_text = topic.get("title", "") + " " + topic.get("query", "")
        query_text = query_text.strip()

        pids_for_topic = topic.get("pids", [])
        qrels_for_topic = qrels_df[qrels_df["topic_id"] == tid]

        relevant_pids = set(
            qrels_for_topic[qrels_for_topic["relevance"] == 1]["doc_id"].tolist()
        )

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
            print(f"Topic {tid}: no docs with abstracts found, skipping")
            continue

        y_true = np.array([1 if pid in relevant_pids else 0 for pid in doc_ids_in_topic])
        n_rel = int(np.sum(y_true))
        print(f"Topic {tid} '{topic.get('title', '')[:60]}': "
              f"{len(doc_ids_in_topic)} docs, {n_rel} relevant")

        if n_rel == 0:
            continue

        scores_dict = {}

        if HAS_BM25:
            scores_dict["bm25"] = run_bm25_ranking(doc_texts_in_topic, query_text, doc_ids_in_topic)

        scores_dict["tfidf"] = run_tfidf_ranking(doc_texts_in_topic, query_text, doc_ids_in_topic)

        scores_dict["minilm"] = run_embedding_ranking(
            doc_texts_in_topic, query_text, doc_ids_in_topic, MINILM_MODEL_NAME,
        )

        scores_dict["specter"] = run_embedding_ranking(
            doc_texts_in_topic, query_text, doc_ids_in_topic, SPECTER_MODEL_NAME,
        )

        for method_name, scores_map in scores_dict.items():
            scores_arr = np.array([scores_map[did] for did in doc_ids_in_topic])
            metrics = evaluate_query(y_true, scores_arr)
            metrics["topic_id"] = tid
            metrics["topic_title"] = topic.get("title", "")
            metrics["method"] = method_name
            metrics["n_docs"] = len(doc_ids_in_topic)
            metrics["n_relevant"] = n_rel
            all_metrics_rows.append(metrics)

        print(f"  Methods: {list(scores_dict.keys())}")

    metrics_df = pd.DataFrame(all_metrics_rows)

    summary_rows = []
    for method in metrics_df["method"].unique():
        method_df = metrics_df[metrics_df["method"] == method]
        row = {"method": method, "n_topics": len(method_df)}

        for col in method_df.columns:
            if col in ("method", "topic_id", "topic_title", "n_topics", "n_docs", "n_relevant"):
                continue
            if method_df[col].dtype in (np.float64, np.int64, float, int):
                row[f"mean_{col}"] = method_df[col].mean()
                row[f"std_{col}"] = method_df[col].std()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "clef_tar_metrics.csv", index=False)
    print(f"\nSaved summary: {OUTPUT_DIR / 'clef_tar_metrics.csv'}")

    display_cols = ["method", "n_topics", "mean_ap", "mean_ndcg_at_10", "mean_ndcg_at_20",
                     "mean_precision_at_100", "mean_recall_at_100"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print("\n=== CLEF-TAR Summary (mean across topics) ===")
    print(summary_df[display_cols].to_string(index=False))

    summary_df.to_csv(OUTPUT_DIR / "clef_tar_table_metrics.csv", index=False)
    print(f"\nSaved table: {OUTPUT_DIR / 'clef_tar_table_metrics.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
