#!/usr/bin/env python3
"""
05c_train_learned_hybrid_reranker.py

Purpose
-------
Train a lightweight learned hybrid reranker for systematic-review triage.

This script takes an existing record-level score table containing:
    - document identifiers
    - reviewer-derived relevance labels
    - ranking/similarity scores from earlier scripts

It trains supervised models to learn a better combination of semantic and lexical
signals than the manually weighted SPECTER-hybrid score.

Recommended use
---------------
Run this after:
    05b_run_additional_embedding_rankers.py

Then rerun:
    06_evaluate_ranking_metrics.py
    09_generate_screening_efficiency_analysis.py

Example
-------
python src/05c_train_learned_hybrid_reranker.py \
    --input outputs/ranking_scores.csv \
    --output outputs/ranking_scores_with_learned_reranker.csv \
    --metrics-output outputs/learned_reranker_metrics.csv

Important
---------
For honest evaluation, this script reports out-of-fold predictions from
cross-validation. It also trains a final model on all records and writes a
full-data score column for future prospective screening.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalise_column_name(name: str) -> str:
    """Make a loose normalised version of a column name for matching."""
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("@", "_at_")
        .replace("/", "_")
    )


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    """
    Find a column from a list of possible names.

    This makes the script more tolerant of your existing naming conventions.
    """
    original_cols = list(df.columns)
    normalised_map = {normalise_column_name(c): c for c in original_cols}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        norm_candidate = normalise_column_name(candidate)
        if norm_candidate in normalised_map:
            return normalised_map[norm_candidate]

    return None


def detect_label_column(df: pd.DataFrame, user_label_col: Optional[str]) -> str:
    """Detect label column or use user-specified label column."""
    if user_label_col:
        if user_label_col not in df.columns:
            raise ValueError(f"Specified label column not found: {user_label_col}")
        return user_label_col

    candidates = [
        "label",
        "relevance_label",
        "is_relevant",
        "included",
        "is_included",
        "final_included",
        "reviewer_label",
        "screening_label_binary",
        "y",
    ]

    label_col = find_first_existing_column(df, candidates)
    if label_col is None:
        raise ValueError(
            "Could not detect label column. Please pass --label-col.\n"
            f"Available columns: {list(df.columns)}"
        )

    return label_col


def convert_labels_to_binary(series: pd.Series) -> pd.Series:
    """
    Convert labels into 0/1.

    Supports:
        - 1 / 0
        - True / False
        - included / excluded / irrelevant
        - Include / Exclude
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        unique_vals = sorted(series.dropna().unique().tolist())
        if set(unique_vals).issubset({0, 1}):
            return series.astype(int)

    positive_values = {
        "1",
        "true",
        "yes",
        "y",
        "included",
        "include",
        "final_included",
        "relevant",
        "positive",
        "accept",
        "accepted",
    }

    negative_values = {
        "0",
        "false",
        "no",
        "n",
        "excluded",
        "exclude",
        "irrelevant",
        "not_relevant",
        "negative",
        "reject",
        "rejected",
    }

    def map_value(x):
        if pd.isna(x):
            return np.nan

        text = str(x).strip().lower()

        if text in positive_values:
            return 1
        if text in negative_values:
            return 0

        raise ValueError(
            f"Could not convert label value to binary: {x!r}. "
            "Please make the label column 1/0 or included/excluded/irrelevant."
        )

    return series.map(map_value).astype(int)


def detect_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    id_cols: Sequence[str],
    user_feature_cols: Optional[List[str]],
) -> List[str]:
    """
    Detect useful feature columns.

    If user passes --feature-cols, use those.
    Otherwise, select numeric columns whose names look like scores/similarities.
    """
    if user_feature_cols:
        missing = [c for c in user_feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        return user_feature_cols

    blocked = set(id_cols) | {label_col}

    feature_keywords = [
        "score",
        "sim",
        "similarity",
        "keyword",
        "bm25",
        "tfidf",
        "tf_idf",
        "minilm",
        "pubmedbert",
        "specter",
        "rq",
        "proposal",
        "embedding",
        "cosine",
        "ranker",
    ]

    features = []
    for col in df.columns:
        if col in blocked:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        norm = normalise_column_name(col)

        if any(key in norm for key in feature_keywords):
            features.append(col)

    if not features:
        # Fallback: all numeric columns except obvious id/label columns.
        features = [
            col for col in df.columns
            if col not in blocked and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not features:
        raise ValueError(
            "No numeric feature columns detected. Please pass --feature-cols."
        )

    return features


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1]
    top_k = order[:k]
    return float(np.mean(y_true[top_k]))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    total_relevant = int(np.sum(y_true))
    if total_relevant == 0:
        return 0.0

    order = np.argsort(scores)[::-1]
    top_k = order[:k]
    return float(np.sum(y_true[top_k]) / total_relevant)


def dcg_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    gains = y_true_sorted[:k]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]

    ideal_order = np.argsort(y_true)[::-1]
    y_ideal = y_true[ideal_order]

    dcg = dcg_at_k(y_sorted, k)
    idcg = dcg_at_k(y_ideal, k)

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def relevant_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> int:
    order = np.argsort(scores)[::-1]
    return int(np.sum(y_true[order[:k]]))


def recovery_depth(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> Optional[int]:
    """
    Minimum rank needed to reach target recall.

    Returns 1-indexed rank.
    """
    total_relevant = int(np.sum(y_true))
    if total_relevant == 0:
        return None

    required = int(np.ceil(total_relevant * target_recall))
    order = np.argsort(scores)[::-1]
    cumulative_relevant = np.cumsum(y_true[order])

    hits = np.where(cumulative_relevant >= required)[0]
    if len(hits) == 0:
        return None

    return int(hits[0] + 1)


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    method_name: str,
    cutoffs: Sequence[int] = (25, 50, 100, 200),
) -> Dict[str, float]:
    """Evaluate ranking scores using systematic-review triage metrics."""
    metrics: Dict[str, float] = {
        "method": method_name,
        "average_precision": float(average_precision_score(y_true, scores)),
        "n_records": int(len(y_true)),
        "n_relevant": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
    }

    prevalence = max(float(np.mean(y_true)), 1e-12)

    for k in cutoffs:
        if k <= len(y_true):
            p = precision_at_k(y_true, scores, k)
            r = recall_at_k(y_true, scores, k)
            n = ndcg_at_k(y_true, scores, k)
            rel = relevant_at_k(y_true, scores, k)

            metrics[f"precision_at_{k}"] = p
            metrics[f"recall_at_{k}"] = r
            metrics[f"ndcg_at_{k}"] = n
            metrics[f"relevant_at_{k}"] = rel
            metrics[f"enrichment_at_{k}"] = float(p / prevalence)

    for target in (0.25, 0.50, 0.75, 0.90):
        rank = recovery_depth(y_true, scores, target)
        label = int(target * 100)
        metrics[f"rank_at_{label}_recall"] = rank if rank is not None else np.nan
        if rank is not None:
            metrics[f"screened_pct_at_{label}_recall"] = float(rank / len(y_true))

    return metrics


def add_augmented_features(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    """
    Add rank-based and agreement features to improve learned reranking.

    Features added:
    - Rank percentile for each score column
    - Top-100 flag for each score column
    - Agreement features: mean, std, max across all scores
    - Lexical vs dense score differences
    - Consensus count: number of methods agreeing in top-100
    """
    out = df.copy()

    for col in score_cols:
        if col not in out.columns:
            continue

        rank_col = f"{col}_rank_pct"
        top_col = f"{col}_top100_flag"

        out[rank_col] = out[col].rank(method="average", pct=True, ascending=True).astype(float)

        order = np.argsort(out[col].values)[::-1]
        top_n = min(100, len(order))
        out[top_col] = 0
        out.iloc[order[:top_n], out.columns.get_loc(top_col)] = 1

    available = [c for c in score_cols if c in out.columns]
    if available:
        score_mat = out[available].astype(float)
        out["score_mean"] = score_mat.mean(axis=1)
        out["score_std"] = score_mat.std(axis=1)
        out["score_max"] = score_mat.max(axis=1)

    lexical = [c for c in ["bm25_score", "tfidf_score", "keyword_score"] if c in out.columns]
    dense = [c for c in ["minilm_score", "specter_score", "medcpt_score",
                         "specter_rq_score", "specter_proposal_score"] if c in out.columns]

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


def make_models() -> Dict[str, Pipeline]:
    """
    Define learned reranker models.

    Logistic regression = interpretable linear learned weighting.
    Ridge-like logistic = more regularized linear model.
    Random forest / ExtraTrees = tree ensembles.
    HistGradientBoosting = nonlinear gradient boosting from sklearn.
    SVM = margin-based reranker.
    MLP = small neural reranker.
    """
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV

    models: Dict[str, Pipeline] = {
        # 1. Main interpretable baseline
        "learned_logistic_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        C=1.0,
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 2. Stronger-regularized logistic
        "learned_logistic_l2_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        penalty="l2",
                        class_weight="balanced",
                        C=0.25,
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 3. Random Forest
        "learned_rf_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        max_depth=6,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 4. Extra Trees, often strong on tabular score features
        "learned_extratrees_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_depth=6,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 5. Sklearn histogram gradient boosting
        "learned_hgb_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 6. Classic Gradient Boosting
        "learned_gb_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        learning_rate=0.035,
                        max_depth=3,
                        min_samples_leaf=5,
                        subsample=0.85,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 7. AdaBoost, sometimes useful for weak feature combinations
        "learned_adaboost_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    AdaBoostClassifier(
                        n_estimators=250,
                        learning_rate=0.04,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 8. RBF SVM, calibrated to probability
        "learned_svm_rbf_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 9. Linear SVM with calibration
        "learned_svm_linear_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            C=0.5,
                            class_weight="balanced",
                            max_iter=10000,
                            random_state=RANDOM_STATE,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),

        # 10. Small neural network reranker
        "learned_mlp_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        alpha=0.01,
                        learning_rate_init=0.001,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # 11. Gaussian Naive Bayes, simple probabilistic baseline
        "learned_nb_specter_triage": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
    }

    return models


def cross_validated_predictions(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    """
    Generate out-of-fold predicted probabilities.

    These are the scores you should use for honest paper evaluation.
    """
    if np.sum(y) < n_splits:
        raise ValueError(
            f"Not enough positive samples for {n_splits}-fold CV. "
            f"Positive samples: {np.sum(y)}"
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_scores = np.zeros(len(y), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y[train_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        if hasattr(fold_model, "predict_proba"):
            scores = fold_model.predict_proba(X_valid)[:, 1]
        else:
            scores = fold_model.decision_function(X_valid)

        oof_scores[valid_idx] = scores

        print(
            f"  Fold {fold}: "
            f"train positives={int(np.sum(y_train))}, "
            f"valid positives={int(np.sum(y[valid_idx]))}"
        )

    return oof_scores


def train_full_model_scores(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[Pipeline, np.ndarray]:
    """
    Train model on all data and return full-data predicted scores.

    These are useful for prospective screening, but not honest evaluation.
    """
    fitted_model = clone(model)
    fitted_model.fit(X, y)

    if hasattr(fitted_model, "predict_proba"):
        scores = fitted_model.predict_proba(X)[:, 1]
    else:
        scores = fitted_model.decision_function(X)

    return fitted_model, scores


def minmax_scale_array(values: np.ndarray) -> np.ndarray:
    """Scale score values to 0-1 for easier comparison/output."""
    values = np.asarray(values).reshape(-1, 1)
    scaler = MinMaxScaler()
    return scaler.fit_transform(values).ravel()


def extract_logistic_coefficients(
    fitted_pipeline: Pipeline,
    feature_cols: List[str],
) -> Optional[pd.DataFrame]:
    """Extract coefficients for logistic regression model."""
    model = fitted_pipeline.named_steps.get("model")

    if not isinstance(model, LogisticRegression):
        return None

    coefs = model.coef_.ravel()

    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return coef_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train learned hybrid reranker for systematic-review triage."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV containing labels and score columns.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV with learned reranker scores added.",
    )

    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional output CSV for metrics.",
    )

    parser.add_argument(
        "--coefficients-output",
        default=None,
        help="Optional output CSV for logistic regression coefficients.",
    )

    parser.add_argument(
        "--label-col",
        default=None,
        help="Name of relevance label column. If omitted, script tries to detect it.",
    )

    parser.add_argument(
        "--id-cols",
        nargs="*",
        default=[
            "record_id",
            "id",
            "title",
            "doi",
            "year",
            "authors",
            "journal",
        ],
        help="Columns to exclude from feature detection.",
    )

    parser.add_argument(
        "--feature-cols",
        nargs="*",
        default=None,
        help=(
            "Explicit feature columns to use. "
            "If omitted, numeric score/similarity columns are detected."
        ),
    )

    parser.add_argument(
        "--manual-hybrid-col",
        default=None,
        help=(
            "Optional name of your existing manual SPECTER-hybrid score column. "
            "If omitted, script tries to detect it."
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )

    parser.add_argument(
        "--primary-model",
        default="learned_logistic_specter_triage",
        choices=[
            "learned_logistic_specter_triage",
            "learned_logistic_l2_specter_triage",
            "learned_rf_specter_triage",
            "learned_extratrees_specter_triage",
            "learned_hgb_specter_triage",
            "learned_gb_specter_triage",
            "learned_adaboost_specter_triage",
            "learned_svm_rbf_specter_triage",
            "learned_svm_linear_specter_triage",
            "learned_mlp_specter_triage",
            "learned_nb_specter_triage",
        ],
        help="Which learned model to expose as the main learned_specter_triage score.",
    )

    parser.add_argument(
        "--cutoffs",
        nargs="*",
        type=int,
        default=[25, 50, 100, 200],
        help="Ranking cutoffs for evaluation.",
    )

    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.metrics_output is None:
        metrics_output_path = output_path.with_name(
            output_path.stem + "_metrics.csv"
        )
    else:
        metrics_output_path = Path(args.metrics_output)

    print(f"Reading input: {input_path}")
    df = pd.read_csv(input_path)

    # Automatically merge TAR TF-IDF+LogReg OOF predictions if available.
    # This gives the learned reranker access to the TAR baseline as a feature,
    # enabling it to learn when to trust TAR's lexical judgement vs. semantic signals.
    tar_col = "tar_tfidf_logreg_score"
    if tar_col not in df.columns:
        tar_path = input_path.parent / "ranking_scores_with_tar_baseline.csv"
        if tar_path.exists():
            tar_df = pd.read_csv(tar_path)
            if tar_col in tar_df.columns:
                merge_cols = ["record_id", tar_col]
                if "record_id" in tar_df.columns and "record_id" in df.columns:
                    df = df.merge(tar_df[merge_cols], on="record_id", how="left")
                    print(f"Merged TAR TF-IDF+LogReg score from {tar_path.name}")
                else:
                    print(f"Warning: Cannot merge TAR score — record_id column missing.")
            else:
                print(f"Warning: {tar_col} not found in {tar_path.name}.")
        else:
            print(f"Note: {tar_path.name} not found. TAR score not available as feature.")

    label_col = detect_label_column(df, args.label_col)
    print(f"Using label column: {label_col}")

    y = convert_labels_to_binary(df[label_col]).to_numpy(dtype=int)

    print(f"Records: {len(df)}")
    print(f"Relevant/included: {int(np.sum(y))}")
    print(f"Prevalence: {np.mean(y):.4f}")

    feature_cols = detect_feature_columns(
        df=df,
        label_col=label_col,
        id_cols=args.id_cols,
        user_feature_cols=args.feature_cols,
    )

    # Avoid leakage from previously produced learned scores if script is rerun.
    feature_cols = [
        c for c in feature_cols
        if not normalise_column_name(c).startswith("learned_")
    ]

    # Add augmented features (rank percentiles, agreement, lexical-dense diffs).
    base_score_cols = feature_cols.copy()
    df = add_augmented_features(df, base_score_cols)

    # Re-detect features to include new augmented columns.
    augmented_feature_cols = [
        c for c in df.columns
        if c not in set(args.id_cols) | {label_col}
        and pd.api.types.is_numeric_dtype(df[c])
        and not normalise_column_name(c).startswith("learned_")
    ]

    print("\nUsing feature columns:")
    for col in augmented_feature_cols:
        print(f"  - {col}")

    X = df[augmented_feature_cols].copy()

    models = make_models()
    metrics_rows: List[Dict[str, float]] = []

    # Evaluate manual hybrid if present.
    manual_hybrid_col = args.manual_hybrid_col
    if manual_hybrid_col is None:
        manual_hybrid_col = find_first_existing_column(
            df,
            [
                "specter_hybrid",
                "specter_hybrid_score",
                "SPECTER-hybrid",
                "hybrid_score",
                "full_hybrid_original",
                "full_hybrid_score",
                "manual_specter_hybrid",
                "manual_hybrid",
            ],
        )

    if manual_hybrid_col is not None and manual_hybrid_col in df.columns:
        print(f"\nDetected manual hybrid score column: {manual_hybrid_col}")
        manual_scores = pd.to_numeric(df[manual_hybrid_col], errors="coerce").fillna(0).to_numpy()
        metrics_rows.append(
            evaluate_scores(
                y_true=y,
                scores=manual_scores,
                method_name="manual_specter_hybrid",
                cutoffs=args.cutoffs,
            )
        )
    else:
        print("\nNo manual hybrid score column detected. Skipping manual hybrid evaluation.")

    # Train and evaluate learned models.
    fitted_models: Dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"\nTraining/evaluating model: {model_name}")

        oof_scores = cross_validated_predictions(
            model=model,
            X=X,
            y=y,
            n_splits=args.n_splits,
        )

        oof_scores_scaled = minmax_scale_array(oof_scores)
        df[f"{model_name}_oof_score"] = oof_scores_scaled

        metrics_rows.append(
            evaluate_scores(
                y_true=y,
                scores=oof_scores_scaled,
                method_name=f"{model_name}_oof",
                cutoffs=args.cutoffs,
            )
        )

        fitted_model, full_scores = train_full_model_scores(model, X, y)
        fitted_models[model_name] = fitted_model

        full_scores_scaled = minmax_scale_array(full_scores)
        df[f"{model_name}_full_score"] = full_scores_scaled

    # Add clean alias columns for the chosen primary model.
    primary = args.primary_model
    df["learned_specter_triage_oof_score"] = df[f"{primary}_oof_score"]
    df["learned_specter_triage_full_score"] = df[f"{primary}_full_score"]

    # For honest reporting in the paper, use OOF.
    df["learned_specter_triage_score"] = df["learned_specter_triage_oof_score"]

    primary_metrics = evaluate_scores(
        y_true=y,
        scores=df["learned_specter_triage_score"].to_numpy(),
        method_name="learned_specter_triage_primary_oof",
        cutoffs=args.cutoffs,
    )
    metrics_rows.append(primary_metrics)

    metrics_df = pd.DataFrame(metrics_rows)

    # Sort output metrics by AP if present.
    if "average_precision" in metrics_df.columns:
        metrics_df = metrics_df.sort_values(
            "average_precision",
            ascending=False,
        )

    print("\nMetrics summary:")
    display_cols = [
        "method",
        "average_precision",
        "precision_at_100",
        "recall_at_100",
        "ndcg_at_100",
        "relevant_at_100",
        "rank_at_50_recall",
        "rank_at_75_recall",
        "rank_at_90_recall",
    ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[display_cols].to_string(index=False))

    # Save outputs.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    metrics_df.to_csv(metrics_output_path, index=False)

    print(f"\nWrote ranked-score table: {output_path}")
    print(f"Wrote metrics table: {metrics_output_path}")

    # Save logistic coefficients if requested.
    if args.coefficients_output:
        coef_output_path = Path(args.coefficients_output)
        logistic_model_name = "learned_logistic_specter_triage"
        logistic_model = fitted_models.get(logistic_model_name)

        if logistic_model is not None:
            coef_df = extract_logistic_coefficients(logistic_model, augmented_feature_cols)
            if coef_df is not None:
                coef_output_path.parent.mkdir(parents=True, exist_ok=True)
                coef_df.to_csv(coef_output_path, index=False)
                print(f"Wrote logistic coefficients: {coef_output_path}")

    # Save a small JSON metadata sidecar.
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "metrics_output": str(metrics_output_path),
        "label_col": label_col,
        "feature_cols": feature_cols,
        "manual_hybrid_col": manual_hybrid_col,
        "primary_model": primary,
        "n_splits": args.n_splits,
        "random_state": RANDOM_STATE,
        "important_note": (
            "Use learned_specter_triage_oof_score for honest retrospective "
            "evaluation. Use learned_specter_triage_full_score only for future "
            "prospective screening."
        ),
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote metadata: {metadata_path}")

    print("\nDone.")
    print(
        "For the extension paper, compare manual_specter_hybrid against "
        "learned_specter_triage_oof_score."
    )


if __name__ == "__main__":
    main()


#     python src/05c_train_learned_hybrid_reranker.py --input outputs/ranking_scores.csv --output outputs/ranking_scores_with_learned_reranker.csv --metrics-output outputs/learned_reranker_metrics.csv --coefficients-output outputs/learned_logistic_coefficients.csv