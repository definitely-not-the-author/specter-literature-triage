#!/usr/bin/env python3
"""
06d_build_clef_tar_ranking_scores.py

Purpose
-------
Build a unified ranking_scores.csv for the CLEF-TAR benchmark,
then train a learned reranker with 5-fold cross-validation.

GPU-aware version:
- Explicitly moves SentenceTransformer models to CUDA if available.
- Explicitly moves MedCPT HuggingFace models and tensors to CUDA if available.
- Uses fewer workers by default because one GPU + many model-loading processes
  can be slower than fewer workers with larger batches.

Usage
-----
python src/06d_build_clef_tar_ranking_scores.py
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, pairwise
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

CLEF_TAR_DIR = Path("data/public_benchmark/clef-tar")
OUTPUT_DIR = Path("outputs/public_benchmark")

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPECTER_MODEL = "allenai-specter"
MEDCPT_QUERY = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE = "ncbi/MedCPT-Article-Encoder"

RANDOM_STATE = 42
N_SPLITS = 5

# IMPORTANT:
# For one GPU, 20 workers is usually inefficient because each worker loads models.
# Start with 4. If GPU memory is too high, use 2. If GPU util is low and memory is fine, try 8.
N_WORKERS = int(os.environ.get("N_WORKERS", "4"))

# Batch sizes
ST_BATCH_SIZE = int(os.environ.get("ST_BATCH_SIZE", "64"))
MEDCPT_BATCH_SIZE = int(os.environ.get("MEDCPT_BATCH_SIZE", "32"))

# Shared status tracking across workers
_status = None
_start_time = None


def _heartbeat(interval=30):
    """Print status every `interval` seconds in a background thread."""
    while True:
        time.sleep(interval)
        if _status is None:
            continue

        done = _status.get("completed", 0)
        total = _status.get("total", 0)
        current_list = list(_status.get("active_topics", []))
        phase = _status.get("phase", "idle")
        elapsed = time.time() - _start_time if _start_time else 0

        log.info(
            f"  [HEARTBEAT] Phase: {phase} | "
            f"Done: {done}/{total} topics | "
            f"Active: {len(current_list)} workers "
            f"({', '.join(current_list[:3])}{'...' if len(current_list) > 3 else ''}) | "
            f"Elapsed: {elapsed:.0f}s"
        )


def parse_topic_file(filepath):
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    topic = {"pids": []}
    current_key = None

    for line in text.split("\n"):
        s = line.strip()

        if s.startswith("Topic:"):
            topic["topic_id"] = s.split(":", 1)[1].strip()

        elif s.startswith("Title:"):
            topic["title"] = s.split(":", 1)[1].strip()

        elif s.startswith("Query:"):
            current_key = "query"
            topic["query"] = s.split(":", 1)[1].strip()

        elif s.startswith("Pids:"):
            current_key = "pids"

        elif current_key == "query" and s:
            topic["query"] = topic.get("query", "") + " " + s

        elif current_key == "pids" and s:
            pid = s.split()[0]
            if pid.isdigit():
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
    p = CLEF_TAR_DIR / "training" / "qrels" / "qrel_abs_train"
    rows = []

    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                rows.append(
                    {
                        "topic_id": parts[0],
                        "doc_id": parts[2],
                        "relevance": int(parts[3]),
                    }
                )

    return pd.DataFrame(rows)


def load_docs():
    cache = CLEF_TAR_DIR / "pubmed_abstracts.csv"
    df = pd.read_csv(cache)

    docs = {}
    for _, row in df.iterrows():
        docs[str(row["pmid"])] = {
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", "")),
        }

    return docs


def tokenize(text):
    return [
        w
        for w in re.sub(r"[^a-z0-9\s-]", " ", str(text).lower()).split()
        if w
    ]


def keyword_score(text):
    text = str(text).lower()

    groups = {
        "clinical": [
            "clinical",
            "trial",
            "patient",
            "treatment",
            "therapy",
            "diagnosis",
        ],
        "systematic": [
            "systematic review",
            "meta-analysis",
            "evidence",
            "cochrane",
        ],
        "methodology": [
            "method",
            "algorithm",
            "model",
            "framework",
            "approach",
        ],
    }

    return float(
        np.mean(
            [
                sum(1 for kw in kws if kw in text) / max(len(kws), 1)
                for kws in groups.values()
            ]
        )
    )


def recovery_depth(y_true, scores, target):
    total = int(np.sum(y_true))

    if total == 0:
        return None

    required = int(np.ceil(total * target))
    order = np.argsort(scores)[::-1]
    cumrel = np.cumsum(y_true[order])
    hits = np.where(cumrel >= required)[0]

    return int(hits[0] + 1) if len(hits) > 0 else len(y_true)


def get_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ── Per-topic worker ──────────────────────────────────────────────

def _process_topic(args):
    """Compute all scores for one topic. Returns list of row dicts."""

    topic_id, topic, docs, qrels_df = args

    from sentence_transformers import SentenceTransformer
    import torch

    device = get_device()

    if _status is not None:
        try:
            active = list(_status.get("active_topics", []))
            active.append(topic_id)
            _status["active_topics"] = active
        except Exception:
            pass

    t0 = time.time()
    log.info(f"[{topic_id}] Start — using device: {device}")
    log.info(f"[{topic_id}] Loading MiniLM + SPECTER")

    st_models = {
        "minilm": SentenceTransformer(MINILM_MODEL, device=device),
        "specter": SentenceTransformer(SPECTER_MODEL, device=device),
    }

    log.info(f"[{topic_id}] MiniLM + SPECTER loaded — loading MedCPT")

    q_tok = AutoTokenizer.from_pretrained(MEDCPT_QUERY)
    q_mdl = AutoModel.from_pretrained(MEDCPT_QUERY).to(device)

    a_tok = AutoTokenizer.from_pretrained(MEDCPT_ARTICLE)
    a_mdl = AutoModel.from_pretrained(MEDCPT_ARTICLE).to(device)

    q_mdl.eval()
    a_mdl.eval()

    log.info(f"[{topic_id}] MedCPT loaded — computing scores")

    query_text = (topic.get("title", "") + " " + topic.get("query", "")).strip()
    pids = topic.get("pids", [])

    qrels_t = qrels_df[qrels_df["topic_id"] == topic_id]
    relevant_pids = set(qrels_t[qrels_t["relevance"] == 1]["doc_id"].tolist())

    doc_ids, doc_texts, y_list = [], [], []
    total_pids = len(pids)

    for i, pid in enumerate(pids):
        if pid in docs:
            d = docs[pid]
            text = f"{d['title']} {d['abstract']}".strip()

            if text:
                doc_ids.append(pid)
                doc_texts.append(text)
                y_list.append(1 if pid in relevant_pids else 0)

        if (i + 1) % 500 == 0 or (i + 1) == total_pids:
            log.info(
                f"[{topic_id}] Loaded {i + 1}/{total_pids} PIDs "
                f"({len(doc_ids)} with abstracts)"
            )

    if not doc_ids or sum(y_list) == 0:
        log.warning(f"[{topic_id}] Skipped — no docs or no relevant")

        if _status is not None:
            try:
                active = list(_status.get("active_topics", []))
                if topic_id in active:
                    active.remove(topic_id)
                _status["active_topics"] = active
                _status["completed"] = _status.get("completed", 0) + 1
            except Exception:
                pass

        return []

    y = np.array(y_list)
    n = len(doc_ids)

    log.info(
        f"[{topic_id}] {n} docs, {int(np.sum(y))} relevant — computing scores"
    )

    # BM25
    tok_docs = [tokenize(t) for t in doc_texts]
    bm25_scores = BM25Okapi(tok_docs).get_scores(tokenize(query_text))
    log.info(f"[{topic_id}] BM25 done")

    # TF-IDF
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    mat = vec.fit_transform(doc_texts + [query_text])
    tfidf_scores = pairwise.cosine_similarity(mat[:-1], mat[-1]).ravel()
    log.info(f"[{topic_id}] TF-IDF done")

    # Sentence Transformers
    st_scores = {}

    for name, model in st_models.items():
        log.info(f"[{topic_id}] Encoding with {name}")

        d_embs = model.encode(
            doc_texts,
            batch_size=ST_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        q_emb = model.encode(
            [query_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        st_scores[name] = pairwise.cosine_similarity(d_embs, q_emb).ravel()
        log.info(f"[{topic_id}] {name} done")

    # MedCPT
    def mean_pool(out, mask):
        embs = out.last_hidden_state
        m = mask.unsqueeze(-1).expand(embs.size()).float()
        return torch.sum(embs * m, dim=1) / torch.clamp(m.sum(dim=1), min=1e-9)

    def enc_batch(tok, mdl, texts, bs=32):
        all_e = []

        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]

                enc = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                enc = {k: v.to(device) for k, v in enc.items()}

                out = mdl(**enc)
                p = mean_pool(out, enc["attention_mask"])
                p = torch.nn.functional.normalize(p, p=2, dim=1)

                all_e.append(p.detach().cpu().numpy())

                if (i + bs) % (bs * 10) == 0 or (i + bs) >= len(texts):
                    log.info(
                        f"[{topic_id}] MedCPT encoded "
                        f"{min(i + bs, len(texts))}/{len(texts)} texts"
                    )

        return np.vstack(all_e)

    log.info(f"[{topic_id}] MedCPT query encoding")
    q_emb = enc_batch(q_tok, q_mdl, [query_text], bs=1)

    log.info(
        f"[{topic_id}] MedCPT article encoding with batch size {MEDCPT_BATCH_SIZE}"
    )
    d_emb = enc_batch(a_tok, a_mdl, doc_texts, bs=MEDCPT_BATCH_SIZE)

    medcpt_scores = pairwise.cosine_similarity(d_emb, q_emb).ravel()
    log.info(f"[{topic_id}] MedCPT done — building rows")

    rows = []

    for i, pid in enumerate(doc_ids):
        rows.append(
            {
                "topic_id": topic_id,
                "record_id": pid,
                "title": docs[pid]["title"],
                "abstract": docs[pid]["abstract"],
                "is_relevant": int(y[i]),
                "keyword_score": keyword_score(doc_texts[i]),
                "bm25_score": float(bm25_scores[i]),
                "tfidf_score": float(tfidf_scores[i]),
                "minilm_score": float(st_scores["minilm"][i]),
                "specter_score": float(st_scores["specter"][i]),
                "medcpt_score": float(medcpt_scores[i]),
            }
        )

    elapsed = time.time() - t0
    log.info(f"[{topic_id}] Done — {len(rows)} rows in {elapsed:.1f}s")

    # Cleanup model memory for this process before returning.
    try:
        del st_models
        del q_tok, q_mdl, a_tok, a_mdl
        del d_emb, q_emb
        gc.collect()

        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    if _status is not None:
        try:
            active = list(_status.get("active_topics", []))
            if topic_id in active:
                active.remove(topic_id)

            _status["active_topics"] = active
            _status["completed"] = _status.get("completed", 0) + 1
        except Exception:
            pass

    return rows


# ── Main ──────────────────────────────────────────────────────────

def main():
    global _start_time, _status

    # spawn is safer with CUDA than fork.
    # fork can work, but CUDA + multiprocessing is usually safer with spawn.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading CLEF-TAR data")
    topics = load_topics()
    qrels_df = load_qrels()
    docs = load_docs()

    log.info(f"  {len(topics)} topics, {len(docs)} cached documents")
    log.info(f"  Detected CPU cores/threads: {mp.cpu_count()}")
    log.info(f"  N_WORKERS: {N_WORKERS}")
    log.info(f"  ST_BATCH_SIZE: {ST_BATCH_SIZE}")
    log.info(f"  MEDCPT_BATCH_SIZE: {MEDCPT_BATCH_SIZE}")

    tasks = [
        (tid, topics[tid], docs, qrels_df)
        for tid in sorted(topics.keys())
    ]

    log.info(
        f"Processing {len(tasks)} topics across {N_WORKERS} workers"
    )
    log.info("Starting workers with 10s stagger to avoid model download/cache collisions")

    _start_time = time.time()

    manager = mp.Manager()
    _status = manager.dict()
    _status["total"] = len(tasks)
    _status["completed"] = 0
    _status["phase"] = "processing"
    _status["active_topics"] = manager.list()

    hb = threading.Thread(target=_heartbeat, args=(30,), daemon=True)
    hb.start()

    with mp.Pool(N_WORKERS) as pool:
        async_results = []

        for i, task in enumerate(tasks):
            ar = pool.apply_async(_process_topic, (task,))
            async_results.append((task[0], ar))

            if i < len(tasks) - 1:
                log.info(
                    f"  Dispatched topic {task[0]} "
                    f"({i + 1}/{len(tasks)}), waiting 10s before next..."
                )
                time.sleep(10)
            else:
                log.info(
                    f"  Dispatched topic {task[0]} "
                    f"({i + 1}/{len(tasks)}) — all topics dispatched"
                )

        log.info("All topics dispatched. Waiting for results...")

        results = []

        for topic_id, ar in async_results:
            try:
                batch = ar.get()
                log.info(f"  Collected {topic_id}: {len(batch)} rows")
                results.append(batch)

            except Exception as e:
                log.exception(f"  ERROR collecting {topic_id}: {e}")
                results.append([])

    all_rows = [row for batch in results for row in batch]

    log.info(
        f"All topics processed in {time.time() - _start_time:.1f}s — "
        f"{len(all_rows)} total rows"
    )

    ranking_df = pd.DataFrame(all_rows)

    if ranking_df.empty:
        raise RuntimeError("No ranking rows were produced. Check input data and worker logs.")

    ranking_scores_path = OUTPUT_DIR / "clef_tar_ranking_scores.csv"
    ranking_df.to_csv(ranking_scores_path, index=False)
    log.info(f"Saved: {ranking_scores_path}")

    # ── Learned reranker with PER-TOPIC 5-fold CV ──────────────────
    # Each topic is trained and evaluated independently to prevent
    # cross-topic label leakage: documents from topic A never appear
    # in the training set when evaluating on topic A's held-out fold.
    log.info("Training learned ExtraTrees reranker (per-topic 5-fold CV)")

    feature_cols = [
        c
        for c in ranking_df.columns
        if c not in (
            "topic_id",
            "record_id",
            "title",
            "abstract",
            "is_relevant",
        )
        and pd.api.types.is_numeric_dtype(ranking_df[c])
    ]

    log.info(f"  Features: {feature_cols}")

    ranking_df["learned_extratrees_oof_score"] = np.nan

    topic_ids_sorted = sorted(ranking_df["topic_id"].unique())

    for topic_id in topic_ids_sorted:
        topic_mask = ranking_df["topic_id"] == topic_id
        topic_indices = np.where(topic_mask)[0]
        topic_df = ranking_df.loc[topic_mask]

        X_topic = topic_df[feature_cols].values
        y_topic = topic_df["is_relevant"].values

        n_rel = int(np.sum(y_topic))
        n_docs = len(y_topic)

        if n_rel < N_SPLITS:
            log.warning(
                f"  [{topic_id}] Skipping — only {n_rel} relevant "
                f"(need >= {N_SPLITS} for {N_SPLITS}-fold CV)"
            )
            ranking_df.loc[topic_mask, "learned_extratrees_oof_score"] = 0.0
            continue

        log.info(
            f"  [{topic_id}] {n_docs} docs, {n_rel} relevant — "
            f"training per-topic 5-fold CV"
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        topic_oof = np.zeros(n_docs, dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_topic, y_topic), start=1):
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_topic[tr_idx])
            X_va = imp.transform(X_topic[va_idx])

            model = ExtraTreesClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

            model.fit(X_tr, y_topic[tr_idx])
            topic_oof[va_idx] = model.predict_proba(X_va)[:, 1]

        ranking_df.loc[topic_mask, "learned_extratrees_oof_score"] = topic_oof

        log.info(
            f"  [{topic_id}] Done — OOF predictions assigned"
        )

    log.info(f"  All {len(topic_ids_sorted)} topics processed")

    # ── TAR TF-IDF+LogReg with PER-TOPIC 5-fold CV ────────────────
    # Same protocol as 05g: TF-IDF (1,2-grams, sublinear_tf, 100K)
    # + LogisticRegression (class-balanced) with 5-fold stratified CV.
    # Trained per-topic to match ExtraTrees protocol (no cross-topic leakage).
    log.info("Training TAR TF-IDF+LogReg baseline (per-topic 5-fold CV)")

    ranking_df["tar_tfidf_logreg_oof_score"] = np.nan

    for topic_id in topic_ids_sorted:
        topic_mask = ranking_df["topic_id"] == topic_id
        topic_df = ranking_df.loc[topic_mask]

        texts = (topic_df["title"].fillna("").astype(str) + " "
                 + topic_df["abstract"].fillna("").astype(str)).str.strip()
        y_topic = topic_df["is_relevant"].values.astype(int)

        n_rel = int(np.sum(y_topic))
        n_docs = len(y_topic)

        if n_rel < N_SPLITS:
            log.warning(
                f"  [{topic_id}] TAR skipped — only {n_rel} relevant "
                f"(need >= {N_SPLITS})"
            )
            ranking_df.loc[topic_mask, "tar_tfidf_logreg_oof_score"] = 0.0
            continue

        log.info(
            f"  [{topic_id}] {n_docs} docs, {n_rel} relevant — "
            f"TAR per-topic 5-fold CV"
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        topic_oof = np.zeros(n_docs, dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(texts.values, y_topic), start=1):
            tar_model = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ])

            tar_model.fit(texts.iloc[tr_idx], y_topic[tr_idx])
            topic_oof[va_idx] = tar_model.predict_proba(texts.iloc[va_idx])[:, 1]

        ranking_df.loc[topic_mask, "tar_tfidf_logreg_oof_score"] = topic_oof

        log.info(f"  [{topic_id}] TAR Done — OOF predictions assigned")

    log.info(f"  All {len(topic_ids_sorted)} TAR topics processed")

    ranking_scores_with_oof_path = OUTPUT_DIR / "clef_tar_ranking_scores_with_oof.csv"
    ranking_df.to_csv(ranking_scores_with_oof_path, index=False)
    log.info(f"Saved: {ranking_scores_with_oof_path}")

    # ── Per-topic evaluation ───────────────────────────────────────
    log.info("Evaluating all methods per topic")

    methods = {
        "bm25_score": "BM25",
        "tfidf_score": "TF-IDF",
        "minilm_score": "MiniLM",
        "specter_score": "SPECTER",
        "medcpt_score": "MedCPT",
        "learned_extratrees_oof_score": "Learned ExtraTrees",
        "tar_tfidf_logreg_oof_score": "TAR TF-IDF+LogReg",
    }

    all_metrics = []

    for topic_id in ranking_df["topic_id"].unique():
        t = ranking_df[ranking_df["topic_id"] == topic_id]
        y_t = t["is_relevant"].values

        if int(np.sum(y_t)) == 0:
            continue

        for score_col, method_name in methods.items():
            scores = t[score_col].values
            ap = average_precision_score(y_t, scores)
            order = np.argsort(scores)[::-1]

            rel100 = (
                int(np.sum(y_t[order[:100]]))
                if len(order) >= 100
                else int(np.sum(y_t[order]))
            )

            all_metrics.append(
                {
                    "topic_id": topic_id,
                    "method": method_name,
                    "average_precision": ap,
                    "relevant_at_100": rel100,
                    "rank_at_50_recall": recovery_depth(y_t, scores, 0.50),
                    "rank_at_75_recall": recovery_depth(y_t, scores, 0.75),
                    "rank_at_90_recall": recovery_depth(y_t, scores, 0.90),
                    "n_docs": len(t),
                    "n_relevant": int(np.sum(y_t)),
                }
            )

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = OUTPUT_DIR / "clef_tar_learned_reranker_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"Saved: {metrics_path}")

    # ── Summary table ──────────────────────────────────────────────
    summary = (
        metrics_df.groupby("method")
        .agg(
            n_topics=("topic_id", "nunique"),
            mean_ap=("average_precision", "mean"),
            std_ap=("average_precision", "std"),
            mean_relevant_at_100=("relevant_at_100", "mean"),
            mean_rank_50=("rank_at_50_recall", "mean"),
            mean_rank_75=("rank_at_75_recall", "mean"),
            mean_rank_90=("rank_at_90_recall", "mean"),
        )
        .reset_index()
        .sort_values("mean_ap", ascending=False)
    )

    summary_path = OUTPUT_DIR / "clef_tar_table_ranking_metrics.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved: {summary_path}")

    print("\n=== CLEF-TAR Summary ===")
    print(summary.to_string(index=False))

    # ── Recovery depth ─────────────────────────────────────────────
    recovery = (
        metrics_df.groupby("method")
        .agg(
            mean_rank_50=("rank_at_50_recall", "mean"),
            mean_rank_75=("rank_at_75_recall", "mean"),
            mean_rank_90=("rank_at_90_recall", "mean"),
        )
        .reset_index()
    )

    recovery_path = OUTPUT_DIR / "clef_tar_table_recovery_depth.csv"
    recovery.to_csv(recovery_path, index=False)
    log.info(f"Saved: {recovery_path}")

    _status["phase"] = "done"
    log.info("Done.")


if __name__ == "__main__":
    main()