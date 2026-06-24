#!/usr/bin/env python3
"""
05e_simulate_active_learning_triage.py

Purpose
-------
Simulate a retrospective active-learning screening workflow.

The simulation mimics a reviewer who screens records in batches,
reveals labels, and uses those labels to train a model that
re-ranks remaining unscreened records.

This demonstrates the human-in-the-loop adaptive capability
of the SPECTER-Triage framework.

Usage
-----
python src/05e_simulate_active_learning_triage.py

Outputs:
  outputs/active_learning_simulation.csv
  figures/active_learning_recall_curve.png
  figures/screening_burden_reduction.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import clone


INPUT_PATH = Path("outputs/ranking_scores_with_learned_reranker.csv")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("outputs/figures")

RANDOM_STATE = 42
BATCH_SIZES = [25, 50, 100]
MAX_ROUNDS = 50
STOPPING_RECALL_IMPROVEMENT_THRESHOLD = 0.01


def compute_recall_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    top_k = order[:k]
    total = int(np.sum(y_true))
    if total == 0:
        return 0.0
    return float(np.sum(y_true[top_k]) / total)


def compute_ap(y_true, scores):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    n_rel = int(np.sum(y_true))
    if n_rel == 0:
        return 0.0
    return float(np.sum(np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1) * y_sorted) / n_rel)


def compute_ndcg_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    ideal_order = np.argsort(y_true)[::-1]
    y_ideal = y_true[ideal_order]

    dcg = float(np.sum(y_sorted[:k] / np.log2(np.arange(2, k + 2))))
    idcg = float(np.sum(y_ideal[:k] / np.log2(np.arange(2, k + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def recovery_depth(y_true, scores, target_recall):
    total = int(np.sum(y_true))
    if total == 0:
        return None
    required = int(np.ceil(total * target_recall))
    order = np.argsort(scores)[::-1]
    cumrel = np.cumsum(y_true[order])
    hits = np.where(cumrel >= required)[0]
    if len(hits) == 0:
        return None
    return int(hits[0] + 1)


def make_model():
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def run_active_learning_simulation(
    df, y_true, feature_cols, initial_rank_col, batch_size,
):
    n = len(df)
    total_relevant = int(np.sum(y_true))

    screened_mask = np.zeros(n, dtype=bool)
    scores = df[initial_rank_col].fillna(0).to_numpy()

    round_log = []
    cumulative_relevant = 0
    prev_recall = 0.0
    stopped_early = False
    stopping_reason = ""

    for round_num in range(1, MAX_ROUNDS + 1):
        order = np.argsort(scores)[::-1]

        unscreened = ~screened_mask
        unscreened_order = order[unscreened[order]]

        batch_indices = unscreened_order[:batch_size]
        if len(batch_indices) == 0:
            break

        new_labels = y_true[batch_indices]
        screened_mask[batch_indices] = True
        cumulative_relevant += int(np.sum(new_labels))

        current_recall = cumulative_relevant / total_relevant if total_relevant > 0 else 0.0
        total_screened = int(np.sum(screened_mask))
        recall_improvement = current_recall - prev_recall

        round_log.append({
            "round": round_num,
            "batch_size": batch_size,
            "total_screened": total_screened,
            "cumulative_relevant": cumulative_relevant,
            "recall": current_recall,
            "recall_improvement": recall_improvement,
            "total_relevant": total_relevant,
            "stopped_early": False,
            "stopping_reason": "",
        })

        if current_recall >= 0.95:
            stopping_reason = "target_recall_reached"
            round_log[-1]["stopping_reason"] = stopping_reason
            break

        if total_screened >= n * 0.8:
            stopping_reason = "collection_exhausted"
            round_log[-1]["stopping_reason"] = stopping_reason
            break

        if round_num >= 3 and recall_improvement < STOPPING_RECALL_IMPROVEMENT_THRESHOLD:
            stopped_early = True
            stopping_reason = "diminishing_returns"
            round_log[-1]["stopped_early"] = True
            round_log[-1]["stopping_reason"] = stopping_reason
            break

        prev_recall = current_recall

        screened_features = df.iloc[screened_mask][feature_cols].to_numpy()
        screened_labels = y_true[screened_mask]

        if len(np.unique(screened_labels)) < 2:
            continue

        try:
            model = make_model()
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(screened_features)
            model.fit(X_imp, screened_labels)

            unscreened_indices = np.where(~screened_mask)[0]
            if len(unscreened_indices) == 0:
                break

            X_unscreened = imp.transform(df.iloc[unscreened_indices][feature_cols].to_numpy())
            new_scores = model.predict_proba(X_unscreened)[:, 1]

            scores[:] = 0
            scores[unscreened_indices] = new_scores

            scores[screened_mask] = -1

        except Exception:
            pass

    if stopped_early:
        print(f"    Stopped early at round {round_num}: {stopping_reason} "
              f"(recall improvement {recall_improvement:.4f} < {STOPPING_RECALL_IMPROVEMENT_THRESHOLD})")

    return pd.DataFrame(round_log)


def compute_benchmark_curves(y_true, score_cols, n):
    benchmark = {}
    for col in score_cols:
        scores = df[col].fillna(0).to_numpy()
        curve = []
        order = np.argsort(scores)[::-1]
        cumrel = 0
        total_rel = int(np.sum(y_true))
        for k in range(1, n + 1):
            if y_true[order[k - 1]]:
                cumrel += 1
            curve.append(cumrel / total_rel if total_rel > 0 else 0)
        benchmark[col] = curve
    return benchmark


if __name__ == "__main__":
    print(f"Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    y_true = df["is_relevant"].to_numpy(dtype=int)
    n = len(df)
    total_relevant = int(np.sum(y_true))
    print(f"Records: {n}, Relevant: {total_relevant}, Prevalence: {total_relevant/n:.4f}")

    feature_cols = [
        c for c in df.columns
        if c not in ("record_id", "title", "doi", "abstract", "screening_label", "is_relevant")
        and df[c].dtype in (np.float64, np.int64, float, int)
        and not c.startswith("learned_")
    ]

    initial_score_cols = ["bm25_score", "minilm_score", "specter_score", "specter_hybrid_score"]
    initial_score_cols = [c for c in initial_score_cols if c in df.columns]

    initial_rank_col = "specter_hybrid_score"
    if initial_rank_col not in df.columns:
        initial_rank_col = initial_score_cols[0]

    print(f"Using initial ranking: {initial_rank_col}")
    print(f"Feature columns: {feature_cols}")
    print(f"Batch sizes: {BATCH_SIZES}\n")

    all_sim_results = []

    for batch_size in BATCH_SIZES:
        print(f"--- Batch size: {batch_size} ---")
        sim_df = run_active_learning_simulation(
            df, y_true, feature_cols, initial_rank_col, batch_size,
        )
        sim_df["batch_size"] = batch_size
        all_sim_results.append(sim_df)

        for target in [0.50, 0.75, 0.90]:
            row = sim_df[sim_df["recall"] >= target]
            if len(row) > 0:
                screened = row.iloc[0]["total_screened"]
                print(f"  Recall@{int(target*100)}: screened {screened}/{n} ({screened/n:.1%})")
            else:
                print(f"  Recall@{int(target*100)}: not reached")

    sim_summary = pd.concat(all_sim_results, ignore_index=True)
    sim_summary.to_csv(OUTPUT_DIR / "active_learning_simulation.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'active_learning_simulation.csv'}")

    print("\n=== Generating figures ===")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for batch_size in BATCH_SIZES:
        bs_df = sim_summary[sim_summary["batch_size"] == batch_size]
        ax.plot(bs_df["total_screened"], bs_df["recall"],
                label=f"Active Learning (batch={batch_size})", linewidth=2)

    random_x = np.arange(1, n + 1)
    random_y = np.full(n, total_relevant / n)
    cumulative_random = np.cumsum(np.random.permutation(y_true)) / total_relevant
    ax.plot(random_x, cumulative_random, "--", color="gray",
            label="Random screening", linewidth=1.5)

    for col in ["specter_hybrid_score", "bm25_score"]:
        if col in df.columns:
            scores = df[col].fillna(0).to_numpy()
            order = np.argsort(scores)[::-1]
            cumrel = np.cumsum(y_true[order]) / total_relevant
            label = col.replace("_score", "").replace("_", " ").title()
            ax.plot(range(1, n + 1), cumrel, ":", label=f"{label} (static)", linewidth=1.5)

    ax.set_xlabel("Records Screened", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Active Learning Simulation: Recall vs Screening Effort", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "active_learning_recall_curve.png", dpi=150)
    plt.savefig(FIGURE_DIR / "active_learning_recall_curve.pdf")
    print(f"Saved: {FIGURE_DIR / 'active_learning_recall_curve.png'}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    methods = []
    r50_vals, r75_vals, r90_vals = [], [], []

    for col in initial_score_cols:
        scores = df[col].fillna(0).to_numpy()
        r50 = recovery_depth(y_true, scores, 0.50)
        r75 = recovery_depth(y_true, scores, 0.75)
        r90 = recovery_depth(y_true, scores, 0.90)
        label = col.replace("_score", "").replace("_", " ").title()
        methods.append(label)
        r50_vals.append(r50 if r50 else n)
        r75_vals.append(r75 if r75 else n)
        r90_vals.append(r90 if r90 else n)

    for batch_size in BATCH_SIZES:
        bs_df = sim_summary[sim_summary["batch_size"] == batch_size]
        r50 = bs_df[bs_df["recall"] >= 0.50]
        r75 = bs_df[bs_df["recall"] >= 0.75]
        r90 = bs_df[bs_df["recall"] >= 0.90]

        r50_vals.append(int(r50.iloc[0]["total_screened"]) if len(r50) > 0 else n)
        r75_vals.append(int(r75.iloc[0]["total_screened"]) if len(r75) > 0 else n)
        r90_vals.append(int(r90.iloc[0]["total_screened"]) if len(r90) > 0 else n)
        methods.append(f"AL (b={batch_size})")

    x = np.arange(len(methods))
    width = 0.25

    ax2.bar(x - width, r50_vals, width, label="50% Recall", color="#2196F3")
    ax2.bar(x, r75_vals, width, label="75% Recall", color="#FF9800")
    ax2.bar(x + width, r90_vals, width, label="90% Recall", color="#F44336")

    ax2.set_ylabel("Records Screened", fontsize=12)
    ax2.set_title("Screening Burden to Reach Target Recall", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "screening_burden_reduction.png", dpi=150)
    plt.savefig(FIGURE_DIR / "screening_burden_reduction.pdf")
    print(f"Saved: {FIGURE_DIR / 'screening_burden_reduction.png'}")

    print("\n=== Summary ===")
    for batch_size in BATCH_SIZES:
        bs_df = sim_summary[sim_summary["batch_size"] == batch_size]
        final_screened = bs_df["total_screened"].max()
        final_recall = bs_df["recall"].max()
        print(f"Batch {batch_size}: screened {final_screened}/{n} ({final_screened/n:.1%}), "
              f"final recall {final_recall:.1%}")

    print("\nDone.")
