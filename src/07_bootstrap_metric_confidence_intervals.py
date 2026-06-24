#!/usr/bin/env python3
"""
07_bootstrap_metric_confidence_intervals.py

Purpose
-------
Compute bootstrap confidence intervals for all ranking metrics to
demonstrate that the learned reranker's improvements are statistically
stable.

Uses paired bootstrap resampling over records: for each bootstrap
iteration, sample N records with replacement, compute metrics on
the sample, and aggregate across iterations.

Usage
-----
python src/07_bootstrap_metric_confidence_intervals.py

Outputs:
  outputs/bootstrap_metric_ci.csv
  outputs/bootstrap_metric_ci_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


INPUT_PATH = Path("outputs/ranking_scores_with_learned_reranker.csv")
AL_PATH = Path("outputs/active_learning_simulation.csv")
CLEF_TAR_METRICS_PATH = Path("outputs/public_benchmark/clef_tar_learned_reranker_metrics.csv")
OUTPUT_DIR = Path("outputs")

N_BOOTSTRAP = 2000
RANDOM_STATE = 42
ALPHA = 0.05


METHODS = {
    "manual_specter_hybrid": "specter_hybrid_score",
    "learned_logistic": "learned_logistic_specter_triage_oof_score",
    "learned_nb": "learned_nb_specter_triage_oof_score",
    "learned_rf": "learned_rf_specter_triage_oof_score",
    "learned_extratrees": "learned_extratrees_specter_triage_oof_score",
    "learned_gb": "learned_gb_specter_triage_oof_score",
    "learned_hgb": "learned_hgb_specter_triage_oof_score",
    "learned_adaboost": "learned_adaboost_specter_triage_oof_score",
    "learned_svm_linear": "learned_svm_linear_specter_triage_oof_score",
    "tar_tfidf_logreg": "tar_tfidf_logreg_score",
}

REFERENCE_METHODS = ["manual_specter_hybrid", "tar_tfidf_logreg"]


def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    top_k = order[:k]
    return float(np.mean(y_true[top_k]))


def recall_at_k(y_true, scores, k):
    total = int(np.sum(y_true))
    if total == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    return float(np.sum(y_true[order[:k]]) / total)


def ndcg_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    ideal_order = np.argsort(y_true)[::-1]
    y_ideal = y_true[ideal_order]

    dcg = float(np.sum(y_sorted[:k] / np.log2(np.arange(2, k + 2))))
    idcg = float(np.sum(y_ideal[:k] / np.log2(np.arange(2, k + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def relevant_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    return int(np.sum(y_true[order[:k]]))


def recovery_depth(y_true, scores, target_recall):
    total = int(np.sum(y_true))
    if total == 0:
        return None
    required = int(np.ceil(total * target_recall))
    order = np.argsort(scores)[::-1]
    cumrel = np.cumsum(y_true[order])
    hits = np.where(cumrel >= required)[0]
    if len(hits) == 0:
        return len(y_true)
    return int(hits[0] + 1)


def compute_all_metrics(y_true, scores):
    metrics = {}
    metrics["ap"] = average_precision_score(y_true, scores)
    metrics["p_at_100"] = precision_at_k(y_true, scores, 100)
    metrics["recall_at_100"] = recall_at_k(y_true, scores, 100)
    metrics["ndcg_at_100"] = ndcg_at_k(y_true, scores, 100)
    metrics["relevant_at_100"] = relevant_at_k(y_true, scores, 100)
    metrics["rank_at_50"] = recovery_depth(y_true, scores, 0.50)
    metrics["rank_at_75"] = recovery_depth(y_true, scores, 0.75)
    metrics["rank_at_90"] = recovery_depth(y_true, scores, 0.90)
    return metrics


def bootstrap_metrics(y_true, scores, n_bootstrap=N_BOOTSTRAP, rng=None):
    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)

    n = len(y_true)
    boot_metrics = {k: [] for k in [
        "ap", "p_at_100", "recall_at_100", "ndcg_at_100",
        "relevant_at_100", "rank_at_50", "rank_at_75", "rank_at_90",
    ]}

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_boot = y_true[indices]
        scores_boot = scores[indices]

        if np.sum(y_boot) == 0:
            continue

        metrics = compute_all_metrics(y_boot, scores_boot)
        for k in boot_metrics:
            boot_metrics[k].append(metrics[k])

    return boot_metrics


def compute_ci(values, alpha=ALPHA):
    arr = np.array(values)
    lower = np.percentile(arr, 100 * alpha / 2)
    upper = np.percentile(arr, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def compute_p_value(boot_diffs):
    arr = np.array(boot_diffs)
    p = np.mean(arr <= 0)
    return float(p)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    y_true = df["is_relevant"].to_numpy(dtype=int)
    n = len(df)
    total_rel = int(np.sum(y_true))
    print(f"Records: {n}, Relevant: {total_rel}")

    available_methods = {}
    for name, col in METHODS.items():
        if col in df.columns:
            available_methods[name] = col
        else:
            print(f"  Warning: column '{col}' not found, skipping {name}")

    print(f"\nMethods: {list(available_methods.keys())}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")

    all_results = []
    boot_cache = {}

    print(f"Precomputing bootstrap for all {len(available_methods)} methods...")
    for method_name, score_col in available_methods.items():
        scores = df[score_col].fillna(0).to_numpy()
        boot_cache[method_name] = bootstrap_metrics(y_true, scores)
        print(f"  Cached: {method_name}")

    for method_name, score_col in available_methods.items():
        print(f"\n--- {method_name} ---")
        boot = boot_cache[method_name]

        for metric_name, values in boot.items():
            if not values:
                continue
            mean_val = float(np.mean(values))
            ci_lower, ci_upper = compute_ci(values)

            row_data = {
                "method": method_name,
                "metric": metric_name,
                "mean": mean_val,
                "ci_lower_95": ci_lower,
                "ci_upper_95": ci_upper,
                "n_bootstrap": len(values),
            }

            for ref_name in REFERENCE_METHODS:
                delta_key = f"delta_vs_{ref_name}"
                p_key = f"p_value_vs_{ref_name}"
                delta = None
                p_value = None
                if method_name != ref_name and ref_name in boot_cache:
                    ref_boot = boot_cache[ref_name]
                    if metric_name in ref_boot and ref_boot[metric_name]:
                        ref_mean = float(np.mean(ref_boot[metric_name]))
                        delta = mean_val - ref_mean
                        diffs = [v - rv for v, rv in zip(values, ref_boot[metric_name])]
                        p_value = compute_p_value(diffs)
                row_data[delta_key] = delta
                row_data[p_key] = p_value

            all_results.append(row_data)

            ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            delta_h = row_data.get("delta_vs_manual_specter_hybrid")
            p_h = row_data.get("p_value_vs_manual_specter_hybrid")
            delta_str = f"{delta_h:+.4f}" if delta_h is not None else "-"
            p_str = f"{p_h:.4f}" if p_h is not None else "-"
            print(f"  {metric_name}: {mean_val:.4f} {ci_str} Δ_hybrid={delta_str} p={p_str}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "bootstrap_metric_ci.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'bootstrap_metric_ci.csv'}")

    summary_cols = ["method", "metric", "mean", "ci_lower_95", "ci_upper_95"]
    for ref_name in REFERENCE_METHODS:
        summary_cols.extend([f"delta_vs_{ref_name}", f"p_value_vs_{ref_name}"])
    summary_cols.append("n_bootstrap")

    summary = results_df[results_df["delta_vs_manual_specter_hybrid"].notna()].copy()
    summary = summary[[c for c in summary_cols if c in summary.columns]]
    summary = summary.sort_values(["metric", "method"])
    summary.to_csv(OUTPUT_DIR / "bootstrap_metric_ci_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'bootstrap_metric_ci_summary.csv'}")

    print("\n=== Key Results: ExtraTrees vs Manual Hybrid ===")
    et_df = results_df[results_df["method"] == "learned_extratrees"]
    for _, row in et_df.iterrows():
        p = row["p_value_vs_manual_specter_hybrid"]
        sig = "***" if p is not None and p < 0.01 else \
              "**" if p is not None and p < 0.05 else \
              "*" if p is not None and p < 0.10 else ""
        delta_str = f"{row['delta_vs_manual_specter_hybrid']:+.4f}" if row["delta_vs_manual_specter_hybrid"] is not None else "-"
        p_str = f"{p:.4f}" if p is not None else "-"
        print(f"  {row['metric']}: {row['mean']:.4f} "
              f"[{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] "
              f"Δ={delta_str} p={p_str} {sig}")

    print("\n=== Key Results: ExtraTrees vs TAR TF-IDF+LogReg ===")
    for _, row in et_df.iterrows():
        p = row.get("p_value_vs_tar_tfidf_logreg")
        sig = "***" if p is not None and p < 0.01 else \
              "**" if p is not None and p < 0.05 else \
              "*" if p is not None and p < 0.10 else ""
        delta = row.get("delta_vs_tar_tfidf_logreg")
        delta_str = f"{delta:+.4f}" if delta is not None else "-"
        p_str = f"{p:.4f}" if p is not None else "-"
        print(f"  {row['metric']}: {row['mean']:.4f} "
              f"[{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] "
              f"Δ={delta_str} p={p_str} {sig}")

    # === Bootstrap Active Learning ===
    if AL_PATH.exists():
        print("\n\n=== Bootstrap Active Learning Screening Depth ===")
        al_df = pd.read_csv(AL_PATH)

        if "initial_method" in al_df.columns:
            al_methods = al_df["initial_method"].unique()
        else:
            al_df["initial_method"] = "specter_hybrid"
            al_methods = ["specter_hybrid"]

        has_topics = "topic_id" in al_df.columns

        al_results = []
        rng = np.random.RandomState(RANDOM_STATE)

        if has_topics:
            for init_method in al_methods:
                for batch_size in al_df["batch_size"].unique():
                    subset = al_df[(al_df["initial_method"] == init_method) &
                                   (al_df["batch_size"] == batch_size)]
                    if subset.empty:
                        continue

                    for target_recall in [0.50, 0.75, 0.90]:
                        screened_vals = []
                        for _ in range(N_BOOTSTRAP):
                            topic_ids = subset["topic_id"].unique()
                            boot_topics = rng.choice(topic_ids, size=len(topic_ids), replace=True)
                            boot_screened = []
                            for tid in boot_topics:
                                t = subset[subset["topic_id"] == tid]
                                if target_recall in t["recall"].values:
                                    row = t[t["recall"] >= target_recall].iloc[0]
                                    boot_screened.append(row["total_screened"])
                            if boot_screened:
                                screened_vals.append(float(np.mean(boot_screened)))

                        if screened_vals:
                            mean_val = float(np.mean(screened_vals))
                            ci_lower, ci_upper = compute_ci(screened_vals)
                            n_docs = subset["n_docs"].iloc[0] if "n_docs" in subset.columns else 2231

                            al_results.append({
                                "initial_method": init_method,
                                "batch_size": batch_size,
                                "target_recall": target_recall,
                                "mean_screened": mean_val,
                                "ci_lower_95": ci_lower,
                                "ci_upper_95": ci_upper,
                                "pct_screened": mean_val / n_docs,
                                "n_bootstrap": len(screened_vals),
                            })

                            print(f"  {init_method} batch={batch_size} "
                                  f"recall={int(target_recall*100)}%: "
                                  f"{mean_val:.0f} [{ci_lower:.0f}, {ci_upper:.0f}] "
                                  f"({mean_val/n_docs:.1%})")
        else:
            print("  Single-topic dataset (no per-topic bootstrap).")
            print("  Active learning results are deterministic.")
            for batch_size in al_df["batch_size"].unique():
                bs_df = al_df[al_df["batch_size"] == batch_size]
                for target_recall in [0.50, 0.75, 0.90]:
                    row = bs_df[bs_df["recall"] >= target_recall]
                    if len(row) > 0:
                        screened = row.iloc[0]["total_screened"]
                        total = bs_df["total_relevant"].iloc[0]
                        n_docs = bs_df["total_screened"].max() + 100
                        print(f"  batch={batch_size}: {int(target_recall*100)}% recall "
                              f"at {int(screened)} records")

        if al_results:
            al_ci_df = pd.DataFrame(al_results)
            al_ci_df.to_csv(OUTPUT_DIR / "bootstrap_active_learning_ci.csv", index=False)
            print(f"\nSaved: {OUTPUT_DIR / 'bootstrap_active_learning_ci.csv'}")

    # === Bootstrap CLEF-TAR External Benchmark ===
    if CLEF_TAR_METRICS_PATH.exists():
        print("\n\n=== Bootstrap CLEF-TAR External Benchmark ===")
        clef_df = pd.read_csv(CLEF_TAR_METRICS_PATH)

        clef_methods = clef_df["method"].unique()
        clef_ci_results = []
        rng = np.random.RandomState(RANDOM_STATE)

        for method in clef_methods:
            method_df = clef_df[clef_df["method"] == method]
            if "topic_id" not in method_df.columns:
                continue

            topic_ids = method_df["topic_id"].unique()

            for metric_col in ["average_precision", "precision_at_100", "recall_at_100", "ndcg_at_100"]:
                if metric_col not in method_df.columns:
                    continue

                boot_vals = []
                for _ in range(N_BOOTSTRAP):
                    boot_topics = rng.choice(topic_ids, size=len(topic_ids), replace=True)
                    boot_metrics = []
                    for tid in boot_topics:
                        t = method_df[method_df["topic_id"] == tid]
                        if not t.empty and metric_col in t.columns:
                            val = t[metric_col].values[0]
                            if not np.isnan(val):
                                boot_metrics.append(val)
                    if boot_metrics:
                        boot_vals.append(float(np.mean(boot_metrics)))

                if boot_vals:
                    mean_val = float(np.mean(boot_vals))
                    ci_lower, ci_upper = compute_ci(boot_vals)

                    clef_ci_results.append({
                        "method": method,
                        "metric": metric_col,
                        "mean": mean_val,
                        "ci_lower_95": ci_lower,
                        "ci_upper_95": ci_upper,
                        "n_bootstrap": len(boot_vals),
                    })

                    print(f"  {method} {metric_col}: {mean_val:.4f} "
                          f"[{ci_lower:.4f}, {ci_upper:.4f}]")

        if clef_ci_results:
            clef_ci_df = pd.DataFrame(clef_ci_results)
            clef_ci_df.to_csv(OUTPUT_DIR / "public_benchmark" / "bootstrap_clef_tar_ci.csv", index=False)
            print(f"\nSaved: {OUTPUT_DIR / 'public_benchmark' / 'bootstrap_clef_tar_ci.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
