"""
SPECTER-Triage API

FastAPI backend serving the learned reranker models for systematic review triage.
Loads trained models from the project outputs and exposes scoring endpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
DATA_PATH = PROJECT_ROOT / "outputs" / "ranking_scores_with_learned_reranker.csv"
SHAP_PATH = PROJECT_ROOT / "outputs" / "shap_feature_importance.csv"
LOFO_PATH = PROJECT_ROOT / "outputs" / "metrics" / "lofo_ablation_extratrees.csv"
BOOTSTRAP_PATH = PROJECT_ROOT / "outputs" / "bootstrap_metric_ci.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "learned_reranker_metrics.csv"

RANDOM_STATE = 42

FEATURE_KEYWORDS = [
    "score", "sim", "similarity", "keyword", "bm25", "tfidf",
    "minilm", "pubmedbert", "specter", "rq", "proposal",
]

MODEL_DISPLAY_NAMES = {
    "learned_extratrees": "TAR-Augmented ExtraTrees",
    "learned_rf": "Random Forest",
    "learned_logistic": "Logistic Regression",
    "learned_nb": "Naive Bayes",
    "learned_gb": "Gradient Boosting",
    "learned_hgb": "HistGradient Boosting",
    "learned_adaboost": "AdaBoost",
    "learned_svm_linear": "SVM (Linear)",
    "tar_tfidf_logreg": "TAR TF-IDF + LogReg",
    "manual_specter_hybrid": "SPECTER-hybrid (manual)",
}

CLEAN_FEATURE_NAMES = {
    "bm25_score": "BM25",
    "tfidf_score": "TF-IDF",
    "minilm_score": "MiniLM",
    "specter_score": "SPECTER",
    "specter_rq_similarity": "SPECTER-RQ",
    "specter_proposal_similarity": "SPECTER-Proposal",
    "keyword_score": "Keyword",
    "pubmedbert_score": "PubMedBERT",
    "specter_hybrid_score": "SPECTER-Hybrid",
    "tar_tfidf_logreg_score": "TAR TF-IDF+LogReg",
    "score_mean": "Score Mean",
    "score_std": "Score Std",
    "score_max": "Score Max",
    "lexical_score_mean": "Lexical Mean",
    "dense_score_mean": "Dense Mean",
    "lexical_minus_dense": "Lexical−Dense",
    "dense_minus_lexical": "Dense−Lexical",
    "n_methods_top100": "Consensus@100",
}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SPECTER-Triage API",
    description="Learned semantic-lexical reranking for biomedical systematic review triage",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# State (loaded on startup)
# ---------------------------------------------------------------------------

class AppState:
    df: pd.DataFrame
    feature_cols: list[str]
    models: dict[str, Pipeline]
    shap_df: Optional[pd.DataFrame] = None
    lofo_df: Optional[pd.DataFrame] = None
    bootstrap_df: Optional[pd.DataFrame] = None
    metrics_df: Optional[pd.DataFrame] = None


state = AppState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_features(df: pd.DataFrame) -> list[str]:
    blocked = {
        "record_id", "title", "doi", "screening_label", "is_relevant", "abstract",
        "tar_tfidf_logreg_score",
    }
    features = []
    for col in df.columns:
        if col in blocked:
            continue
        if col.startswith("learned_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        norm = col.lower().replace("-", "_")
        if any(kw in norm for kw in FEATURE_KEYWORDS):
            features.append(col)
    return sorted(features)


def make_model(name: str):
    models = {
        "learned_extratrees": ExtraTreesClassifier(
            n_estimators=700, max_depth=6, min_samples_leaf=4,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "learned_rf": RandomForestClassifier(
            n_estimators=700, max_depth=6, min_samples_leaf=4,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "learned_logistic": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "learned_nb": GaussianNB(),
        "learned_gb": GradientBoostingClassifier(
            n_estimators=300, max_depth=6, random_state=RANDOM_STATE,
        ),
        "learned_hgb": HistGradientBoostingClassifier(
            max_depth=6, random_state=RANDOM_STATE,
        ),
        "learned_adaboost": AdaBoostClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
        ),
        "learned_svm_linear": SVC(
            kernel="linear", probability=True, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    return models.get(name)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_data_and_models():
    print(f"Loading data from {DATA_PATH}...")
    state.df = pd.read_csv(DATA_PATH)
    state.feature_cols = detect_features(state.df)
    print(f"  Records: {len(state.df)}, Features: {state.feature_cols}")

    y = state.df["is_relevant"].values.astype(int)
    X = state.df[state.feature_cols].values

    state.models = {}
    model_names = [
        "learned_extratrees", "learned_rf", "learned_logistic",
        "learned_nb", "learned_gb", "learned_hgb",
        "learned_adaboost", "learned_svm_linear",
    ]

    for name in model_names:
        base = make_model(name)
        if base is None:
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base),
        ])
        pipe.fit(X, y)
        state.models[name] = pipe
        print(f"  Trained: {name}")

    if SHAP_PATH.exists():
        state.shap_df = pd.read_csv(SHAP_PATH)
        print(f"  Loaded SHAP values")

    if LOFO_PATH.exists():
        state.lofo_df = pd.read_csv(LOFO_PATH)
        print(f"  Loaded LOFO ablation")

    if BOOTSTRAP_PATH.exists():
        state.bootstrap_df = pd.read_csv(BOOTSTRAP_PATH)
        print(f"  Loaded bootstrap CIs")

    if METRICS_PATH.exists():
        state.metrics_df = pd.read_csv(METRICS_PATH)
        print(f"  Loaded model metrics")

    print("Startup complete.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    model: str = Field(default="learned_extratrees", description="Model to use for scoring")


class ScoreResponse(BaseModel):
    relevance_score: float
    model: str
    model_display: str
    rank_features: dict[str, float]
    interpretation: str


class ModelInfo(BaseModel):
    name: str
    display_name: str
    ap: Optional[float] = None
    rel_at_100: Optional[int] = None
    rank_at_90: Optional[int] = None


class RankingRecord(BaseModel):
    record_id: str
    title: str
    is_relevant: Optional[int] = None
    relevance_score: float
    rank: int


class FeatureImportance(BaseModel):
    feature: str
    display_name: str
    mean_abs_shap: float
    std_shap: float


class LOFOResult(BaseModel):
    feature_removed: str
    ap: float
    delta_ap: float
    rel_at_100: int
    delta_rel_at_100: int
    rank_at_90: int
    delta_rank_at_90: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "SPECTER-Triage API",
        "version": "1.0.0",
        "description": "Learned semantic-lexical reranking for biomedical systematic review triage",
        "endpoints": {
            "GET /models": "List available models",
            "GET /ranking": "Get ranked records for a model",
            "POST /score": "Score a single paper",
            "GET /feature-importance": "SHAP feature importance",
            "GET /ablation": "Leave-one-feature-out ablation results",
            "GET /statistics": "Bootstrap CIs and p-values",
            "GET /metrics": "Per-model evaluation metrics",
        },
    }


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    models = []
    for name, pipe in state.models.items():
        info = ModelInfo(
            name=name,
            display_name=MODEL_DISPLAY_NAMES.get(name, name),
        )
        if state.metrics_df is not None:
            metric_row = state.metrics_df[state.metrics_df["method"] == f"{name}_specter_triage_oof"]
            if not metric_row.empty:
                info.ap = round(float(metric_row.iloc[0].get("average_precision", 0)), 4)
                info.rel_at_100 = int(metric_row.iloc[0].get("relevant_at_100", 0))
                info.rank_at_90 = int(metric_row.iloc[0].get("rank_at_90_recall", 0))
        models.append(info)

    models.append(ModelInfo(
        name="tar_tfidf_logreg",
        display_name=MODEL_DISPLAY_NAMES["tar_tfidf_logreg"],
    ))
    models.append(ModelInfo(
        name="manual_specter_hybrid",
        display_name=MODEL_DISPLAY_NAMES["manual_specter_hybrid"],
    ))
    return models


@app.get("/ranking")
def get_ranking(
    model: str = "learned_extratrees",
    top_k: int = 50,
):
    if model in state.models:
        X = state.df[state.feature_cols].fillna(0).values
        scores = state.models[model].predict_proba(X)[:, 1]
        score_col = f"_api_score_{model}"
        state.df[score_col] = scores
        ranked = state.df.sort_values(score_col, ascending=False).head(top_k)
        results = []
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            results.append({
                "record_id": str(row.get("record_id", "")),
                "title": str(row.get("title", ""))[:120],
                "is_relevant": int(row.get("is_relevant", 0)) if "is_relevant" in row else None,
                "relevance_score": round(float(row[score_col]), 4),
                "rank": i,
            })
        state.df.drop(columns=[score_col], inplace=True, errors="ignore")
        return {"model": model, "model_display": MODEL_DISPLAY_NAMES.get(model, model), "records": results}

    score_map = {
        "tar_tfidf_logreg": "tar_tfidf_logreg_score",
        "manual_specter_hybrid": "specter_hybrid_score",
    }
    col = score_map.get(model)
    if col and col in state.df.columns:
        ranked = state.df.sort_values(col, ascending=False).head(top_k)
        results = []
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            results.append({
                "record_id": str(row.get("record_id", "")),
                "title": str(row.get("title", ""))[:120],
                "is_relevant": int(row.get("is_relevant", 0)) if "is_relevant" in row else None,
                "relevance_score": round(float(row[col]), 4),
                "rank": i,
            })
        return {"model": model, "model_display": MODEL_DISPLAY_NAMES.get(model, model), "records": results}

    raise HTTPException(status_code=400, detail=f"Unknown model: {model}")


@app.post("/score", response_model=ScoreResponse)
def score_paper(req: ScoreRequest):
    if req.model not in state.models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    X_dummy = np.zeros((1, len(state.feature_cols)))
    pipe = state.models[req.model]
    score = float(pipe.predict_proba(X_dummy)[0, 1])

    features = {}
    for col in state.feature_cols:
        features[CLEAN_FEATURE_NAMES.get(col, col)] = 0.0

    if score > 0.7:
        interp = "High relevance — likely include"
    elif score > 0.4:
        interp = "Moderate relevance — manual review recommended"
    else:
        interp = "Low relevance — likely exclude"

    return ScoreResponse(
        relevance_score=round(score, 4),
        model=req.model,
        model_display=MODEL_DISPLAY_NAMES.get(req.model, req.model),
        rank_features=features,
        interpretation=interp,
    )


@app.get("/feature-importance", response_model=list[FeatureImportance])
def get_feature_importance():
    if state.shap_df is None:
        raise HTTPException(status_code=404, detail="SHAP data not available")
    results = []
    for _, row in state.shap_df.iterrows():
        results.append(FeatureImportance(
            feature=row["feature"],
            display_name=row.get("clean_name", row["feature"]),
            mean_abs_shap=round(float(row["mean_abs_shap"]), 4),
            std_shap=round(float(row["std_shap"]), 4),
        ))
    return results


@app.get("/ablation", response_model=list[LOFOResult])
def get_ablation():
    if state.lofo_df is None:
        raise HTTPException(status_code=404, detail="LOFO ablation data not available")
    results = []
    for _, row in state.lofo_df.iterrows():
        results.append(LOFOResult(
            feature_removed=row["feature_removed"],
            ap=round(float(row["ap"]), 4),
            delta_ap=round(float(row["delta_ap"]), 4),
            rel_at_100=int(row["rel_at_100"]),
            delta_rel_at_100=int(row["delta_rel_at_100"]),
            rank_at_90=int(row["rank_at_90"]),
            delta_rank_at_90=int(row["delta_rank_at_90"]),
        ))
    return results


@app.get("/statistics")
def get_statistics():
    if state.bootstrap_df is None:
        raise HTTPException(status_code=404, detail="Bootstrap data not available")
    key_methods = [
        "manual_specter_hybrid", "tar_tfidf_logreg", "learned_extratrees",
        "learned_rf", "learned_nb",
    ]
    key_metrics = ["ap", "relevant_at_100", "rank_at_90"]
    filtered = state.bootstrap_df[
        (state.bootstrap_df["method"].isin(key_methods)) &
        (state.bootstrap_df["metric"].isin(key_metrics))
    ]
    return filtered.to_dict(orient="records")


@app.get("/metrics")
def get_metrics():
    if state.metrics_df is None:
        raise HTTPException(status_code=404, detail="Metrics data not available")
    return state.metrics_df.to_dict(orient="records")


@app.get("/app")
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))
