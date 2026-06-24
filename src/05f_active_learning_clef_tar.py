#!/usr/bin/env python3
"""
05f_active_learning_clef_tar.py

Purpose
-------
Run the retrospective active-learning simulation on the CLEF eHealth TAR
benchmark to demonstrate that the reviewer-in-the-loop adaptation
generalises beyond the main mutational-signature review dataset.

For each Cochrane review topic, we simulate screening in batches,
retraining the model on revealed labels and re-ranking remaining records.

Usage
-----
Prerequisites:
  - Run 06b to cache PubMed abstracts and compute baseline rankings

python src/05f_active_learning_clef_tar.py

Outputs:
  outputs/public_benchmark/clef_tar_active_learning.csv
  figures/clef_tar_active_learning_recall_curve.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import clone

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

import torch
from sentence_transformers import SentenceTransformer


CLEF_TAR_DIR = Path("data/public_benchmark/clef-tar")
OUTPUT_DIR = Path("outputs/public_benchmark")
FIGURE_DIR = Path("outputs/figures")

RANDOM_STATE = 42
BATCH_SIZES = [25, 50, 100]
MAX_ROUNDS = 100
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER_MODEL_NAME = "allenai-specter"


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


def load_topics():
    topics_dir = CLEF_TAR_DIR / "training" / "topics_train"
    topics = {}
    for fp in sorted(topics_dir.iterdir()):
        if fp.is_file() and not fp.name.startswith("."):
            t = parse_topic_file(fp)
            if "topic_id" in t:
                topics[t["topic_id"]] = t
    return topics


def load_qrels():
    qrels_dir = CLEF_TAR_DIR / "training" / "qrels"
    preferred = ["qrel_abs_train", "train.abs.qrels"]
    qrels_path = None
    for name in preferred:
        p = qrels_dir / name
        if p.exists():
            qrels_path = p
            break
    if qrels_path is None:
        qrels_path = sorted(qrels_dir.iterdir())[0]

    rows = []
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                rows.append({"topic_id": parts[0], "doc_id": parts[2], "relevance": int(parts[3])})
    return pd.DataFrame(rows)


def load_cached_abstracts():
    cache_path = CLEF_TAR_DIR / "pubmed_abstracts.csv"
    cache_df = pd.read_csv(cache_path)
    docs = {}
    for _, row in cache_df.iterrows():
        docs[str(row["pmid"])] = {
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", "")),
        }
    return docs


def compute_scores(doc_texts, query_text, doc_ids, models):
    scores = {}

    if HAS_BM25:
        import re
        def tokenize(t):
            t = re.sub(r"[^a-z0-9\s-]", " ", str(t).lower())
            return [w for w in t.split() if w]
        tok_docs = [tokenize(t) for t in doc_texts]
        tok_q = tokenize(query_text)
        bm25 = BM25Okapi(tok_docs)
        scores["bm25"] = dict(zip(doc_ids, bm25.get_scores(tok_q)))

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(doc_texts + [query_text])
    tfidf_scores = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    scores["tfidf"] = dict(zip(doc_ids, tfidf_scores))

    for model_name, model in models.items():
        doc_embs = model.encode(doc_texts, batch_size=32, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
        q_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        emb_scores = cosine_similarity(doc_embs, q_emb).ravel()
        scores[model_name] = dict(zip(doc_ids, emb_scores))

    return scores


def simulate_active_learning(y_true, initial_scores, feature_matrix, batch_size):
    n = len(y_true)
    total_rel = int(np.sum(y_true))
    if total_rel == 0:
        return pd.DataFrame()

    scores = initial_scores.copy()
    screened = np.zeros(n, dtype=bool)
    cumrel = 0
    rows = []

    for rnd in range(1, MAX_ROUNDS + 1):
        order = np.argsort(scores)[::-1]
        unscreened_order = order[~screened[order]]
        batch_idx = unscreened_order[:batch_size]
        if len(batch_idx) == 0:
            break

        cumrel += int(np.sum(y_true[batch_idx]))
        screened[batch_idx] = True
        recall = cumrel / total_rel
        rows.append({"round": rnd, "total_screened": int(np.sum(screened)),
                      "cumulative_relevant": cumrel, "recall": recall})

        if recall >= 0.95 or np.sum(screened) >= n * 0.8:
            break

        if len(np.unique(y_true[screened])) < 2:
            continue

        try:
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(feature_matrix[screened])
            model = ExtraTreesClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=4,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
            )
            model.fit(X_tr, y_true[screened])

            unscreened_idx = np.where(~screened)[0]
            X_te = imp.transform(feature_matrix[unscreened_idx])
            new_scores = model.predict_proba(X_te)[:, 1]

            scores = np.zeros(n)
            scores[unscreened_idx] = new_scores
            scores[screened] = -1
        except Exception:
            pass

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CLEF-TAR data...")
    topics = load_topics()
    qrels_df = load_qrels()
    docs = load_cached_abstracts()
    print(f"  {len(topics)} topics, {len(docs)} cached documents")

    print("Loading embedding models...")
    models = {
        "minilm": SentenceTransformer(MINILM_MODEL_NAME),
        "specter": SentenceTransformer(SPECTER_MODEL_NAME),
    }
    print("  Models loaded.")

    all_results = []

    for tid, topic in sorted(topics.items()):
        query_text = (topic.get("title", "") + " " + topic.get("query", "")).strip()
        pids = topic.get("pids", [])
        qrels_t = qrels_df[qrels_df["topic_id"] == tid]
        relevant_pids = set(qrels_t[qrels_t["relevance"] == 1]["doc_id"].tolist())

        doc_ids, doc_texts, y_list = [], [], []
        for pid in pids:
            if pid in docs:
                d = docs[pid]
                text = f"{d['title']} {d['abstract']}".strip()
                if text:
                    doc_ids.append(pid)
                    doc_texts.append(text)
                    y_list.append(1 if pid in relevant_pids else 0)

        if not doc_ids or sum(y_list) == 0:
            continue

        y_true = np.array(y_list)
        n = len(doc_ids)
        n_rel = int(np.sum(y_true))

        print(f"\nTopic {tid}: {n} docs, {n_rel} relevant")

        scores_dict = compute_scores(doc_texts, query_text, doc_ids, models)

        all_score_keys = list(scores_dict.keys())
        feature_matrix = np.column_stack([scores_dict[m][d] for d in doc_ids
                                           for m in all_score_keys
                                           if m in scores_dict and d in scores_dict[m]])
        if feature_matrix.shape[1] == 0:
            continue

        for batch_size in BATCH_SIZES:
            for init_method in ["bm25", "tfidf", "minilm", "specter"]:
                if init_method not in scores_dict:
                    continue

                initial_scores = np.array([scores_dict[init_method].get(d, 0) for d in doc_ids])
                sim_df = simulate_active_learning(y_true, initial_scores, feature_matrix, batch_size)

                if sim_df.empty:
                    continue

                sim_df["topic_id"] = tid
                sim_df["batch_size"] = batch_size
                sim_df["initial_method"] = init_method
                sim_df["n_docs"] = n
                sim_df["n_relevant"] = n_rel
                all_results.append(sim_df)

                for target in [0.50, 0.75, 0.90]:
                    row = sim_df[sim_df["recall"] >= target]
                    if len(row) > 0:
                        s = row.iloc[0]["total_screened"]
                        print(f"  init={init_method}, batch={batch_size}: {int(target*100)}% recall at {s}/{n} ({s/n:.1%})")

    if not all_results:
        print("\nNo results. Check that topics have abstracts and relevant docs.")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(OUTPUT_DIR / "clef_tar_active_learning.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'clef_tar_active_learning.csv'}")

    print("\n=== Generating recall curve ===")
    fig, ax = plt.subplots(figsize=(10, 6))

    for init_method in ["bm25", "minilm", "specter"]:
        for bs in BATCH_SIZES:
            subset = results_df[(results_df["initial_method"] == init_method) & (results_df["batch_size"] == bs)]
            if subset.empty:
                continue
            avg = subset.groupby("total_screened")["recall"].mean().reset_index()
            label = f"{init_method.upper()} init (batch={bs})"
            linestyle = "-" if bs == 50 else "--" if bs == 25 else ":"
            ax.plot(avg["total_screened"], avg["recall"],
                    label=label, linewidth=2, linestyle=linestyle)

    random_x = np.arange(1, results_df["n_docs"].max() + 1)
    random_y = np.cumsum(np.random.RandomState(42).permutation(
        np.concatenate([np.ones(20), np.zeros(2000)]))) / 20
    ax.plot(random_x[:len(random_y)], random_y, "--", color="gray",
            label="Random screening", linewidth=1.5)

    ax.set_xlabel("Records Screened (mean across topics)", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("CLEF-TAR: Active Learning Recall Curve", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "clef_tar_active_learning_recall_curve.png", dpi=150)
    plt.savefig(FIGURE_DIR / "clef_tar_active_learning_recall_curve.pdf")
    print(f"Saved: {FIGURE_DIR / 'clef_tar_active_learning_recall_curve.png'}")

    print("\n=== Summary (mean across topics) ===")
    summary = results_df.groupby(["initial_method", "batch_size"]).agg(
        mean_screened_50=("total_screened", lambda x: x[results_df.loc[x.index, "recall"] >= 0.50].min() if (results_df.loc[x.index, "recall"] >= 0.50).any() else None),
        mean_screened_90=("total_screened", lambda x: x[results_df.loc[x.index, "recall"] >= 0.90].min() if (results_df.loc[x.index, "recall"] >= 0.90).any() else None),
    ).reset_index()
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
