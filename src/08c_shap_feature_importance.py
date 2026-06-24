#!/usr/bin/env python3
"""
08c_shap_feature_importance.py

Purpose
-------
Compute SHAP feature importance for the TAR-Augmented ExtraTrees reranker
to explain which retrieval signals contribute most to predictions.

Requires: pip install shap

Usage
-----
python src/08c_shap_feature_importance.py

Outputs:
  outputs/shap_feature_importance.csv
  outputs/figures/shap_summary.png
  outputs/figures/shap_summary.pdf
  outputs/figures/shap_bar.png
  outputs/figures/shap_bar.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


RANDOM_STATE = 42
SHAP_SUBSAMPLE = 500
TALL_FIGSIZE = (10, 7.4)

FEATURE_KEYWORDS = [
    "score", "sim", "similarity", "keyword", "bm25", "tfidf",
    "minilm", "pubmedbert", "specter", "rq", "proposal",
]

CLEAN_NAMES = {
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
    "medcpt_score": "MedCPT",
}


def detect_features(df: pd.DataFrame) -> list[str]:
    """Detect feature columns by keyword matching."""
    blocked = {
        "record_id", "title", "doi", "screening_label", "is_relevant", "abstract",
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


def main():
    if not HAS_SHAP:
        print("ERROR: shap library not installed. Run: pip install shap")
        return

    parser = argparse.ArgumentParser(description="SHAP feature importance for TAR-Augmented ExtraTrees")
    parser.add_argument("--input", default="outputs/ranking_scores_with_learned_reranker.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--subsample", type=int, default=SHAP_SUBSAMPLE)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)
    feature_cols = detect_features(df)
    print(f"Detected {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols].values
    y = df["is_relevant"].values.astype(int)

    print("Training ExtraTrees on full data...")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=700, max_depth=6, min_samples_leaf=4,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X, y)

    X_imputed = pipe.named_steps["imputer"].transform(X)
    clf = pipe.named_steps["clf"]

    n_subsample = min(args.subsample, len(X_imputed))
    rng = np.random.RandomState(RANDOM_STATE)
    subsample_idx = rng.choice(len(X_imputed), size=n_subsample, replace=False)
    X_sub = X_imputed[subsample_idx]

    print(f"Computing SHAP values on {n_subsample} samples...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sub)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    std_shap = np.std(shap_vals, axis=0)

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "clean_name": [CLEAN_NAMES.get(f, f) for f in feature_cols],
        "mean_abs_shap": mean_abs_shap,
        "std_shap": std_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    print(f"\nSaved: {output_dir / 'shap_feature_importance.csv'}")
    print("\nFeature importance ranking:")
    print(shap_df.to_string(index=False))

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    fig, ax = plt.subplots(figsize=TALL_FIGSIZE)
    sorted_df = shap_df.sort_values("mean_abs_shap")
    ax.barh(sorted_df["clean_name"], sorted_df["mean_abs_shap"], color="#0072B2")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance: TAR-Augmented ExtraTrees")
    fig.tight_layout()
    fig.savefig(figures_dir / "shap_bar.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / "shap_bar.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {figures_dir / 'shap_bar.png'}")

    feature_names = [CLEAN_NAMES.get(f, f) for f in feature_cols]
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sub, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title("SHAP Summary: TAR-Augmented ExtraTrees")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=600, bbox_inches="tight")
    plt.savefig(figures_dir / "shap_summary.pdf", bbox_inches="tight")
    plt.close("all")
    print(f"Saved: {figures_dir / 'shap_summary.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
