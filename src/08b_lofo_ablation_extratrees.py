#!/usr/bin/env python3
"""
08b_lofo_ablation_extratrees.py

Purpose
-------
Leave-one-feature-out (LOFO) ablation of the TAR-Augmented ExtraTrees reranker.
For each feature, retrain the model excluding that feature and measure
the impact on AP, Rel@100, and Rank@90%.

Usage
-----
python src/08b_lofo_ablation_extratrees.py

Outputs:
  outputs/metrics/lofo_ablation_extratrees.csv
  outputs/tables/table_lofo_ablation.csv
  outputs/figures/lofo_ablation_ap.png
  outputs/figures/lofo_ablation_ap.pdf
  outputs/figures/lofo_ablation_rank90.png
  outputs/figures/lofo_ablation_rank90.pdf
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
N_BOOTSTRAP = 2000
ALPHA = 0.05
TALL_FIGSIZE = (10, 7.4)

FEATURE_KEYWORDS = [
    "score", "sim", "similarity", "keyword", "bm25", "tfidf",
    "minilm", "pubmedbert", "specter", "rq", "proposal",
]

COLORS = {
    "gain": "#009E73",
    "loss": "#D55E00",
    "neutral": "#999999",
}


def detect_features(df: pd.DataFrame) -> list[str]:
    """Detect feature columns by keyword matching, excluding learned/label/id cols."""
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


def train_evaluate_oof(df: pd.DataFrame, feature_cols: list[str],
                       y_col: str = "is_relevant") -> dict:
    """Train ExtraTrees with 5-fold OOF and return metrics."""
    X = df[feature_cols].values
    y = df[y_col].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=700, max_depth=6, min_samples_leaf=4,
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ])
        pipe.fit(X_train, y_train)
        oof_scores[val_idx] = pipe.predict_proba(X_val)[:, 1]

    ap = average_precision_score(y, oof_scores)
    order = np.argsort(oof_scores)[::-1]
    rel_at_100 = int(np.sum(y[order[:100]]))
    total_rel = int(np.sum(y))
    required = int(np.ceil(total_rel * 0.90))
    cumrel = np.cumsum(y[order])
    hits = np.where(cumrel >= required)[0]
    rank_90 = int(hits[0] + 1) if len(hits) > 0 else len(y)

    return {"ap": ap, "rel_at_100": rel_at_100, "rank_at_90": rank_90}


def bootstrap_delta(y_true, scores_a, scores_b, metric_fn, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap the difference between two score vectors."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if np.sum(y_b) == 0:
            continue
        va = metric_fn(y_b, scores_a[idx])
        vb = metric_fn(y_b, scores_b[idx])
        if va is not None and vb is not None:
            diffs.append(va - vb)
    if not diffs:
        return None, None, None
    arr = np.array(diffs)
    return float(np.mean(arr)), float(np.percentile(arr, 100 * ALPHA / 2)), float(np.percentile(arr, 100 * (1 - ALPHA / 2)))


def main():
    parser = argparse.ArgumentParser(description="LOFO ablation of ExtraTrees reranker")
    parser.add_argument("--input", default="outputs/ranking_scores_with_learned_reranker.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for d in [metrics_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)
    feature_cols = detect_features(df)
    print(f"Detected {len(feature_cols)} features: {feature_cols}")

    y = df["is_relevant"].values.astype(int)

    print("\n=== Baseline (all features) ===")
    baseline = train_evaluate_oof(df, feature_cols)
    print(f"  AP={baseline['ap']:.4f}  Rel@100={baseline['rel_at_100']}  Rank@90%={baseline['rank_at_90']}")

    results = []
    for feat in feature_cols:
        reduced = [f for f in feature_cols if f != feat]
        print(f"\n=== Dropping: {feat} ({len(reduced)} features remain) ===")
        metrics = train_evaluate_oof(df, reduced)
        delta_ap = baseline["ap"] - metrics["ap"]
        delta_rel = baseline["rel_at_100"] - metrics["rel_at_100"]
        delta_rank = metrics["rank_at_90"] - baseline["rank_at_90"]
        print(f"  AP={metrics['ap']:.4f} (Δ={delta_ap:+.4f})  "
              f"Rel@100={metrics['rel_at_100']} (Δ={delta_rel:+d})  "
              f"Rank@90%={metrics['rank_at_90']} (Δ={delta_rank:+d})")
        results.append({
            "feature_removed": feat,
            "ap": metrics["ap"],
            "delta_ap": delta_ap,
            "rel_at_100": metrics["rel_at_100"],
            "delta_rel_at_100": delta_rel,
            "rank_at_90": metrics["rank_at_90"],
            "delta_rank_at_90": delta_rank,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("delta_ap", ascending=False)

    results_df.to_csv(metrics_dir / "lofo_ablation_extratrees.csv", index=False)
    print(f"\nSaved: {metrics_dir / 'lofo_ablation_extratrees.csv'}")

    table = results_df.copy()
    table.insert(0, "baseline_ap", f"{baseline['ap']:.4f}")
    table.insert(1, "baseline_rel100", baseline["rel_at_100"])
    table.insert(2, "baseline_rank90", baseline["rank_at_90"])
    table.to_csv(tables_dir / "table_lofo_ablation.csv", index=False)
    print(f"Saved: {tables_dir / 'table_lofo_ablation.csv'}")

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    fig, ax = plt.subplots(figsize=TALL_FIGSIZE)
    sorted_df = results_df.sort_values("delta_ap")
    colors = [COLORS["loss"] if v > 0 else COLORS["gain"] if v < 0 else COLORS["neutral"]
              for v in sorted_df["delta_ap"]]
    ax.barh(sorted_df["feature_removed"], sorted_df["delta_ap"], color=colors)
    ax.set_xlabel("ΔAP (drop = positive = feature was helpful)")
    ax.set_title("Leave-One-Feature-Out Ablation: Average Precision")
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "lofo_ablation_ap.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / "lofo_ablation_ap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {figures_dir / 'lofo_ablation_ap.png'}")

    fig, ax = plt.subplots(figsize=TALL_FIGSIZE)
    sorted_df = results_df.sort_values("delta_rank_at_90")
    colors = [COLORS["loss"] if v > 0 else COLORS["gain"] if v < 0 else COLORS["neutral"]
              for v in sorted_df["delta_rank_at_90"]]
    ax.barh(sorted_df["feature_removed"], sorted_df["delta_rank_at_90"], color=colors)
    ax.set_xlabel("ΔRank@90% (positive = feature was helpful, rank increased)")
    ax.set_title("Leave-One-Feature-Out Ablation: Rank@90%")
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "lofo_ablation_rank90.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / "lofo_ablation_rank90.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {figures_dir / 'lofo_ablation_rank90.png'}")

    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
