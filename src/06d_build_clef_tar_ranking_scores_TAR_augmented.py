#!/usr/bin/env python3
"""
06d_build_clef_tar_ranking_scores.py

Purpose
-------
Build a unified ranking_scores.csv for the CLEF-TAR benchmark, then evaluate:

1. Standalone retrieval baselines:
   - BM25
   - TF-IDF query similarity
   - MiniLM
   - SPECTER
   - MedCPT

2. TAR TF-IDF+LogReg baseline:
   - Per-topic stratified 5-fold out-of-fold prediction
   - No cross-topic leakage

3. Plain learned ExtraTrees reranker:
   - Uses retrieval-score features only
   - Per-topic stratified 5-fold out-of-fold prediction
   - No cross-topic leakage

4. TAR-augmented ExtraTrees reranker:
   - Uses retrieval-score features PLUS fold-wise TAR TF-IDF+LogReg score
   - Strict outer-fold stacking:
       For each topic and each outer fold:
         a. Train TAR on the outer training fold only
         b. Score outer training and validation records
         c. Train ExtraTrees on retrieval features + TAR score for outer training
         d. Predict outer validation records
   - This avoids using labels from the held-out fold when producing the stacked
     ExtraTrees prediction for that fold.

GPU-aware version:
- Explicitly moves SentenceTransformer models to CUDA if available.
- Explicitly moves MedCPT HuggingFace models and tensors to CUDA if available.
- Uses fewer workers by default because one GPU + many model-loading processes
  can be slower than fewer workers with larger batches.

Usage
-----
python src/06d_build_clef_tar_ranking_scores.py

Recommended on one GPU:
N_WORKERS=1 ST_BATCH_SIZE=64 MEDCPT_BATCH_SIZE=32 python src/06d_build_clef_tar_ranking_scores.py

Outputs
-------
outputs/public_benchmark/clef_tar_ranking_scores.csv
outputs/public_benchmark/clef_tar_ranking_scores_with_oof.csv
outputs/public_benchmark/clef_tar_learned_reranker_metrics.csv
outputs/public_benchmark/clef_tar_table_ranking_metrics.csv
outputs/public_benchmark/clef_tar_table_recovery_depth.csv
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
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------
# Logging and constants
# ---------------------------------------------------------------------

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
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))

# IMPORTANT:
# For one GPU, 20 workers is usually inefficient because each worker loads models.
# Start with 1 or 2. If GPU memory is fine and util is low, try 4.
N_WORKERS = int(os.environ.get("N_WORKERS", "4"))

# Batch sizes
ST_BATCH_SIZE = int(os.environ.get("ST_BATCH_SIZE", "64"))
MEDCPT_BATCH_SIZE = int(os.environ.get("MEDCPT_BATCH_SIZE", "32"))

# Shared status tracking across workers
_status = None
_start_time = None


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _heartbeat(interval: int = 30):
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


def parse_topic_file(filepath: Path) -> Dict:
    text = filepath.read_text(encoding="utf-8", errors="replace")
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


def load_topics() -> Dict[str, Dict]:
    topics_dir = CLEF_TAR_DIR / "training" / "topics_train"
    topics = {}

    for fp in sorted(topics_dir.iterdir()):
        if fp.is_file() and not fp.name.startswith("."):
            t = parse_topic_file(fp)
            if "topic_id" in t:
                topics[t["topic_id"]] = t

    return topics


def load_qrels() -> pd.DataFrame:
    p = CLEF_TAR_DIR / "training" / "qrels" / "qrel_abs_train"
    rows = []

    with open(p, encoding="utf-8", errors="replace") as f:
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


def load_docs() -> Dict[str, Dict[str, str]]:
    cache = CLEF_TAR_DIR / "pubmed_abstracts.csv"
    df = pd.read_csv(cache)

    docs = {}
    for _, row in df.iterrows():
        docs[str(row["pmid"])] = {
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", "")),
        }

    return docs


def tokenize(text: str) -> List[str]:
    return [
        w
        for w in re.sub(r"[^a-z0-9\s-]", " ", str(text).lower()).split()
        if w
    ]


def keyword_score(text: str) -> float:
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


def recovery_depth(y_true: np.ndarray, scores: np.ndarray, target: float) -> Optional[int]:
    total = int(np.sum(y_true))

    if total == 0:
        return None

    required = int(np.ceil(total * target))
    order = np.argsort(scores)[::-1]
    cumrel = np.cumsum(y_true[order])
    hits = np.where(cumrel >= required)[0]

    return int(hits[0] + 1) if len(hits) > 0 else len(y_true)


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def make_tar_pipeline() -> Pipeline:
    """TAR TF-IDF+LogReg baseline used consistently across CLEF topics."""
    return Pipeline(
        [
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
        ]
    )


def make_extratrees() -> ExtraTreesClassifier:
    """Primary tabular learned reranker."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def add_within_topic_rank_features(
    df: pd.DataFrame,
    base_score_cols: List[str],
) -> pd.DataFrame:
    """
    Add rank and percentile-rank features within each topic.

    These are non-label features and are safe because they only use the score
    distribution inside the candidate set for that topic.
    """
    out = df.copy()

    for col in base_score_cols:
        if col not in out.columns:
            continue

        rank_col = f"{col}_rank_pct"
        top_col = f"{col}_top100_flag"

        # Higher scores should have higher percentiles.
        out[rank_col] = (
            out.groupby("topic_id")[col]
            .rank(method="average", pct=True, ascending=True)
            .astype(float)
        )

        # Top-100 flag within topic.
        out[top_col] = 0
        for topic_id, idx in out.groupby("topic_id").groups.items():
            scores = out.loc[idx, col].values
            order = np.argsort(scores)[::-1]
            top_n = min(100, len(order))
            top_idx = np.array(list(idx))[order[:top_n]]
            out.loc[top_idx, top_col] = 1

    # Agreement features.
    available = [c for c in base_score_cols if c in out.columns]
    if available:
        score_mat = out[available].astype(float)
        out["score_mean"] = score_mat.mean(axis=1)
        out["score_std"] = score_mat.std(axis=1)
        out["score_max"] = score_mat.max(axis=1)

    lexical = [c for c in ["bm25_score", "tfidf_score"] if c in out.columns]
    dense = [c for c in ["minilm_score", "specter_score", "medcpt_score"] if c in out.columns]

    if lexical:
        out["lexical_score_mean"] = out[lexical].astype(float).mean(axis=1)
    if dense:
        out["dense_score_mean"] = out[dense].astype(float).mean(axis=1)
    if lexical and dense:
        out["lexical_minus_dense"] = out["lexical_score_mean"] - out["dense_score_mean"]
        out["dense_minus_lexical"] = out["dense_score_mean"] - out["lexical_score_mean"]

    top_flags = [f"{c}_top100_flag" for c in available if f"{c}_top100_flag" in out.columns]
    if top_flags:
        out["n_methods_top100"] = out[top_flags].sum(axis=1)

    return out


# ---------------------------------------------------------------------
# Per-topic worker for retrieval feature generation
# ---------------------------------------------------------------------

def _process_topic(args):
    """Compute all retrieval scores for one topic. Returns list of row dicts."""

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

    # TF-IDF query similarity
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    mat = vec.fit_transform(doc_texts + [query_text])
    tfidf_scores = pairwise.cosine_similarity(mat[:-1], mat[-1]).ravel()
    log.info(f"[{topic_id}] TF-IDF query similarity done")

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
        title = docs[pid]["title"]
        abstract = docs[pid]["abstract"]
        text = doc_texts[i]

        rows.append(
            {
                "topic_id": topic_id,
                "record_id": pid,
                "title": title,
                "abstract": abstract,
                "is_relevant": int(y[i]),
                "keyword_score": keyword_score(text),
                "bm25_score": float(bm25_scores[i]),
                "tfidf_score": float(tfidf_scores[i]),
                "minilm_score": float(st_scores["minilm"][i]),
                "specter_score": float(st_scores["specter"][i]),
                "medcpt_score": float(medcpt_scores[i]),
                "title_len": len(str(title).split()),
                "abstract_len": len(str(abstract).split()),
                "text_len": len(str(text).split()),
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


# ---------------------------------------------------------------------
# Supervised per-topic models
# ---------------------------------------------------------------------

def compute_tar_oof_per_topic(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TAR TF-IDF+LogReg out-of-fold scores per topic.

    This is the standalone TAR baseline. It is trained and evaluated within
    each topic only, preventing cross-topic leakage.
    """
    log.info("Training TAR TF-IDF+LogReg baseline (per-topic 5-fold CV)")

    ranking_df = ranking_df.copy()
    ranking_df["tar_tfidf_logreg_oof_score"] = np.nan

    topic_ids_sorted = sorted(ranking_df["topic_id"].unique())

    for topic_id in topic_ids_sorted:
        topic_mask = ranking_df["topic_id"] == topic_id
        topic_df = ranking_df.loc[topic_mask].copy()

        texts = (
            topic_df["title"].fillna("").astype(str)
            + " "
            + topic_df["abstract"].fillna("").astype(str)
        ).str.strip()

        y_topic = topic_df["is_relevant"].values.astype(int)

        n_rel = int(np.sum(y_topic))
        n_docs = len(y_topic)

        if n_rel < N_SPLITS or (n_docs - n_rel) < N_SPLITS:
            log.warning(
                f"  [{topic_id}] TAR skipped — positives={n_rel}, "
                f"negatives={n_docs - n_rel}; need >= {N_SPLITS} per class"
            )
            ranking_df.loc[topic_mask, "tar_tfidf_logreg_oof_score"] = 0.0
            continue

        log.info(
            f"  [{topic_id}] {n_docs} docs, {n_rel} relevant — "
            f"TAR per-topic {N_SPLITS}-fold CV"
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        topic_oof = np.zeros(n_docs, dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(texts.values, y_topic), start=1):
            tar_model = make_tar_pipeline()
            tar_model.fit(texts.iloc[tr_idx], y_topic[tr_idx])
            topic_oof[va_idx] = tar_model.predict_proba(texts.iloc[va_idx])[:, 1]

            log.info(
                f"    [{topic_id}] TAR fold {fold}: "
                f"train positives={int(np.sum(y_topic[tr_idx]))}, "
                f"valid positives={int(np.sum(y_topic[va_idx]))}"
            )

        ranking_df.loc[topic_mask, "tar_tfidf_logreg_oof_score"] = topic_oof
        log.info(f"  [{topic_id}] TAR done — OOF predictions assigned")

    log.info(f"  All {len(topic_ids_sorted)} TAR topics processed")
    return ranking_df


def compute_plain_extratrees_oof_per_topic(
    ranking_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute plain ExtraTrees OOF scores using retrieval/tabular features only.

    This is useful as a non-TAR-augmented learned reranker baseline.
    """
    log.info("Training plain learned ExtraTrees reranker (per-topic 5-fold CV)")

    ranking_df = ranking_df.copy()
    ranking_df["learned_extratrees_oof_score"] = np.nan

    topic_ids_sorted = sorted(ranking_df["topic_id"].unique())

    for topic_id in topic_ids_sorted:
        topic_mask = ranking_df["topic_id"] == topic_id
        topic_df = ranking_df.loc[topic_mask].copy()

        X_topic = topic_df[feature_cols].values
        y_topic = topic_df["is_relevant"].values.astype(int)

        n_rel = int(np.sum(y_topic))
        n_docs = len(y_topic)

        if n_rel < N_SPLITS or (n_docs - n_rel) < N_SPLITS:
            log.warning(
                f"  [{topic_id}] Plain ET skipped — positives={n_rel}, "
                f"negatives={n_docs - n_rel}; need >= {N_SPLITS} per class"
            )
            ranking_df.loc[topic_mask, "learned_extratrees_oof_score"] = 0.0
            continue

        log.info(
            f"  [{topic_id}] {n_docs} docs, {n_rel} relevant — "
            f"plain ExtraTrees per-topic {N_SPLITS}-fold CV"
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

            model = make_extratrees()
            model.fit(X_tr, y_topic[tr_idx])
            topic_oof[va_idx] = model.predict_proba(X_va)[:, 1]

            log.info(
                f"    [{topic_id}] Plain ET fold {fold}: "
                f"train positives={int(np.sum(y_topic[tr_idx]))}, "
                f"valid positives={int(np.sum(y_topic[va_idx]))}"
            )

        ranking_df.loc[topic_mask, "learned_extratrees_oof_score"] = topic_oof
        log.info(f"  [{topic_id}] Plain ET done — OOF predictions assigned")

    log.info(f"  All {len(topic_ids_sorted)} plain ET topics processed")
    return ranking_df


def compute_tar_augmented_extratrees_oof_per_topic(
    ranking_df: pd.DataFrame,
    retrieval_feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute TAR-augmented ExtraTrees OOF scores using strict outer-fold stacking.

    For each topic and each outer fold:
    - TAR is trained only on the outer training fold.
    - TAR scores are generated for the outer training and validation folds.
    - ExtraTrees is trained on retrieval features + TAR score for the training fold.
    - ExtraTrees predicts the validation fold.

    Therefore, the validation prediction does not use labels from the validation
    fold in either the TAR component or the ExtraTrees component.
    """
    log.info("Training TAR-augmented ExtraTrees reranker (strict per-topic outer-fold stacking)")

    ranking_df = ranking_df.copy()
    ranking_df["tar_augmented_extratrees_oof_score"] = np.nan

    topic_ids_sorted = sorted(ranking_df["topic_id"].unique())

    for topic_id in topic_ids_sorted:
        topic_mask = ranking_df["topic_id"] == topic_id
        topic_df = ranking_df.loc[topic_mask].copy()

        texts = (
            topic_df["title"].fillna("").astype(str)
            + " "
            + topic_df["abstract"].fillna("").astype(str)
        ).str.strip()

        X_retrieval = topic_df[retrieval_feature_cols].values
        y_topic = topic_df["is_relevant"].values.astype(int)

        n_rel = int(np.sum(y_topic))
        n_docs = len(y_topic)

        if n_rel < N_SPLITS or (n_docs - n_rel) < N_SPLITS:
            log.warning(
                f"  [{topic_id}] TAR-aug ET skipped — positives={n_rel}, "
                f"negatives={n_docs - n_rel}; need >= {N_SPLITS} per class"
            )
            ranking_df.loc[topic_mask, "tar_augmented_extratrees_oof_score"] = 0.0
            continue

        log.info(
            f"  [{topic_id}] {n_docs} docs, {n_rel} relevant — "
            f"TAR-augmented ExtraTrees per-topic {N_SPLITS}-fold CV"
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        topic_oof = np.zeros(n_docs, dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_retrieval, y_topic), start=1):
            # 1) Train TAR only on the outer training fold.
            tar_model = make_tar_pipeline()
            tar_model.fit(texts.iloc[tr_idx], y_topic[tr_idx])

            # 2) Produce fold-local TAR scores for train and validation.
            tar_tr = tar_model.predict_proba(texts.iloc[tr_idx])[:, 1]
            tar_va = tar_model.predict_proba(texts.iloc[va_idx])[:, 1]

            # 3) Add TAR score as an additional tabular feature.
            X_tr_aug = np.column_stack([X_retrieval[tr_idx], tar_tr])
            X_va_aug = np.column_stack([X_retrieval[va_idx], tar_va])

            # 4) Train ExtraTrees on augmented training features.
            imp = SimpleImputer(strategy="median")
            X_tr_aug = imp.fit_transform(X_tr_aug)
            X_va_aug = imp.transform(X_va_aug)

            model = make_extratrees()
            model.fit(X_tr_aug, y_topic[tr_idx])
            topic_oof[va_idx] = model.predict_proba(X_va_aug)[:, 1]

            log.info(
                f"    [{topic_id}] TAR-aug ET fold {fold}: "
                f"train positives={int(np.sum(y_topic[tr_idx]))}, "
                f"valid positives={int(np.sum(y_topic[va_idx]))}"
            )

        ranking_df.loc[topic_mask, "tar_augmented_extratrees_oof_score"] = topic_oof
        log.info(f"  [{topic_id}] TAR-augmented ET done — OOF predictions assigned")

    log.info(f"  All {len(topic_ids_sorted)} TAR-augmented ET topics processed")
    return ranking_df


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_methods_per_topic(ranking_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Evaluating all methods per topic")

    methods = {
        "bm25_score": "BM25",
        "tfidf_score": "TF-IDF",
        "minilm_score": "MiniLM",
        "specter_score": "SPECTER",
        "medcpt_score": "MedCPT",
        "tar_tfidf_logreg_oof_score": "TAR TF-IDF+LogReg",
        "learned_extratrees_oof_score": "Learned ExtraTrees",
        "tar_augmented_extratrees_oof_score": "TAR-Augmented ExtraTrees",
    }

    all_metrics = []

    for topic_id in sorted(ranking_df["topic_id"].unique()):
        t = ranking_df[ranking_df["topic_id"] == topic_id]
        y_t = t["is_relevant"].values.astype(int)

        if int(np.sum(y_t)) == 0:
            continue

        for score_col, method_name in methods.items():
            if score_col not in t.columns:
                log.warning(f"Skipping missing score column: {score_col}")
                continue

            scores = pd.to_numeric(t[score_col], errors="coerce").fillna(0).values

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
                    "average_precision": float(ap),
                    "relevant_at_100": rel100,
                    "rank_at_50_recall": recovery_depth(y_t, scores, 0.50),
                    "rank_at_75_recall": recovery_depth(y_t, scores, 0.75),
                    "rank_at_90_recall": recovery_depth(y_t, scores, 0.90),
                    "n_docs": len(t),
                    "n_relevant": int(np.sum(y_t)),
                }
            )

    return pd.DataFrame(all_metrics)


def save_summary_tables(metrics_df: pd.DataFrame) -> None:
    metrics_path = OUTPUT_DIR / "clef_tar_learned_reranker_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"Saved: {metrics_path}")

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global _start_time, _status

    # spawn is safer with CUDA than fork.
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
    log.info(f"  N_SPLITS: {N_SPLITS}")

    tasks = [
        (tid, topics[tid], docs, qrels_df)
        for tid in sorted(topics.keys())
    ]

    log.info(f"Processing {len(tasks)} topics across {N_WORKERS} workers")
    log.info("Starting workers with 10s stagger to avoid model download/cache collisions")

    _start_time = time.time()

    manager = mp.Manager()
    _status = manager.dict()
    _status["total"] = len(tasks)
    _status["completed"] = 0
    _status["phase"] = "processing retrieval features"
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

    # Add non-label rank/agreement features.
    base_score_cols = [
        "keyword_score",
        "bm25_score",
        "tfidf_score",
        "minilm_score",
        "specter_score",
        "medcpt_score",
    ]
    ranking_df = add_within_topic_rank_features(ranking_df, base_score_cols)

    ranking_scores_path = OUTPUT_DIR / "clef_tar_ranking_scores.csv"
    ranking_df.to_csv(ranking_scores_path, index=False)
    log.info(f"Saved: {ranking_scores_path}")

    # 1) TAR baseline first.
    _status["phase"] = "TAR TF-IDF+LogReg"
    ranking_df = compute_tar_oof_per_topic(ranking_df)

    # Feature columns for plain and TAR-augmented ExtraTrees.
    # Do not include labels, IDs, text, or supervised output columns.
    excluded = {
        "topic_id",
        "record_id",
        "title",
        "abstract",
        "is_relevant",
        "tar_tfidf_logreg_oof_score",
        "learned_extratrees_oof_score",
        "tar_augmented_extratrees_oof_score",
    }

    retrieval_feature_cols = [
        c
        for c in ranking_df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(ranking_df[c])
    ]

    log.info("Retrieval/tabular features for ExtraTrees:")
    for c in retrieval_feature_cols:
        log.info(f"  - {c}")

    # 2) Plain ExtraTrees without TAR as a feature.
    _status["phase"] = "plain ExtraTrees"
    ranking_df = compute_plain_extratrees_oof_per_topic(
        ranking_df=ranking_df,
        feature_cols=retrieval_feature_cols,
    )

    # 3) TAR-augmented ExtraTrees with strict fold-wise TAR stacking.
    _status["phase"] = "TAR-augmented ExtraTrees"
    ranking_df = compute_tar_augmented_extratrees_oof_per_topic(
        ranking_df=ranking_df,
        retrieval_feature_cols=retrieval_feature_cols,
    )

    # Save final score table.
    ranking_scores_with_oof_path = OUTPUT_DIR / "clef_tar_ranking_scores_with_oof.csv"
    ranking_df.to_csv(ranking_scores_with_oof_path, index=False)
    log.info(f"Saved: {ranking_scores_with_oof_path}")

    # Evaluate.
    _status["phase"] = "evaluation"
    metrics_df = evaluate_methods_per_topic(ranking_df)
    save_summary_tables(metrics_df)

    _status["phase"] = "done"
    log.info("Done.")


if __name__ == "__main__":
    main()
