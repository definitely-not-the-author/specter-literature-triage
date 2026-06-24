#!/usr/bin/env python3
"""
07b_clef_tar_paired_bootstrap.py

Purpose
-------
Perform paired bootstrap resampling across CLEF-TAR topics to determine
whether TAR-Augmented ExtraTrees consistently outperforms TAR (TF-IDF+LogReg)
across diverse systematic review topics.

This provides external validity evidence: if ExtraTrees beats TAR on
14-15 of 20 CLEF topics, that's both statistically significant AND
generalizable.

Usage
-----
python src/07b_clef_tar_paired_bootstrap.py

Outputs:
  outputs/public_benchmark/clef_tar_per_topic_comparison.csv
  outputs/public_benchmark/clef_tar_paired_bootstrap_summary.csv
  outputs/public_benchmark/clef_tar_paired_bootstrap_ci.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("outputs/public_benchmark/clef_tar_learned_reranker_metrics.csv")
OUTPUT_DIR = Path("outputs/public_benchmark")

N_BOOTSTRAP = 10000
RANDOM_STATE = 42
ALPHA = 0.05


def compute_ci(values, alpha=ALPHA):
    """Compute confidence interval."""
    arr = np.array(values)
    lower = np.percentile(arr, 100 * alpha / 2)
    upper = np.percentile(arr, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading per-topic data: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # Get unique topics
    topics = df["topic_id"].unique()
    print(f"Number of CLEF-TAR topics: {len(topics)}")

    # Extract AP for each method per topic
    methods_to_compare = ["TAR-Augmented ExtraTrees", "TAR TF-IDF+LogReg"]
    method_data = {m: [] for m in methods_to_compare}

    for topic in topics:
        topic_df = df[df["topic_id"] == topic]
        for method in methods_to_compare:
            method_row = topic_df[topic_df["method"] == method]
            if not method_row.empty and "average_precision" in method_row.columns:
                method_data[method].append(method_row["average_precision"].values[0])
            else:
                method_data[method].append(np.nan)

    # Convert to arrays
    et_ap = np.array(method_data["TAR-Augmented ExtraTrees"])
    tar_ap = np.array(method_data["TAR TF-IDF+LogReg"])

    # Remove topics where either method is NaN
    valid_mask = ~(np.isnan(et_ap) | np.isnan(tar_ap))
    et_ap = et_ap[valid_mask]
    tar_ap = tar_ap[valid_mask]
    valid_topics = topics[valid_mask]

    print(f"\nValid topics for comparison: {len(et_ap)}")
    print(f"ExtraTrees AP range: [{et_ap.min():.4f}, {et_ap.max():.4f}]")
    print(f"TAR TF-IDF AP range: [{tar_ap.min():.4f}, {tar_ap.max():.4f}]")

    # Paired bootstrap
    print(f"\nRunning paired bootstrap ({N_BOOTSTRAP} iterations)...")
    rng = np.random.RandomState(RANDOM_STATE)

    boot_diffs = []
    boot_win_rates = []

    for _ in range(N_BOOTSTRAP):
        # Sample topics with replacement
        indices = rng.choice(len(valid_topics), size=len(valid_topics), replace=True)

        boot_et = et_ap[indices]
        boot_tar = tar_ap[indices]

        boot_diffs.append(np.mean(boot_et) - np.mean(boot_tar))
        boot_win_rates.append(np.sum(boot_et > boot_tar) / len(indices))

    boot_diffs = np.array(boot_diffs)
    boot_win_rates = np.array(boot_win_rates)

    # Compute statistics
    mean_diff = np.mean(boot_diffs)
    ci_lower, ci_upper = compute_ci(boot_diffs)
    raw_p = np.mean(boot_diffs <= 0)
    p_value = max(raw_p, 1.0 / N_BOOTSTRAP)  # floor at 1/N_BOOTSTRAP

    mean_win_rate = np.mean(boot_win_rates)
    win_rate_ci_lower, win_rate_ci_upper = compute_ci(boot_win_rates)

    # Win count per topic
    et_win_count = int(np.sum(et_ap > tar_ap))
    tar_win_count = int(np.sum(tar_ap > et_ap))
    ties = int(np.sum(et_ap == tar_ap))

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: TAR-Augmented ExtraTrees vs TAR TF-IDF across CLEF-TAR Topics")
    print("=" * 70)

    print(f"\nPer-Topic Statistics:")
    print(f"  ExtraTrees AP: {np.mean(et_ap):.4f} ± {np.std(et_ap):.4f}")
    print(f"  TAR TF-IDF AP: {np.mean(tar_ap):.4f} ± {np.std(tar_ap):.4f}")

    print(f"\nPaired Bootstrap Results:")
    print(f"  Mean difference (ET - TAR): {mean_diff:+.4f}")
    print(f"  95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    if raw_p == 0:
        print(f"  p-value: < {1.0/N_BOOTSTRAP:.4f} (0 of {N_BOOTSTRAP} bootstrap samples favoured TAR)")
    else:
        print(f"  p-value: {p_value:.4f}")
    

    sig_level = ""
    if p_value < 0.01:
        sig_level = "***"
    elif p_value < 0.05:
        sig_level = "**"
    elif p_value < 0.10:
        sig_level = "*"
    else:
        sig_level = "ns"

    print(f"  Significance: {sig_level}")

    print(f"\nWin Count:")
    print(f"  ExtraTrees wins: {et_win_count}/{len(et_ap)} topics")
    print(f"  TAR wins: {tar_win_count}/{len(et_ap)} topics")
    print(f"  Ties: {ties}/{len(et_ap)} topics")

    print(f"\nWin Rate:")
    print(f"  Mean win rate: {mean_win_rate:.1%}")
    print(f"  95% CI: [{win_rate_ci_lower:.1%}, {win_rate_ci_upper:.1%}]")

    # Create per-topic comparison DataFrame
    comparison_data = []
    for i, topic in enumerate(valid_topics):
        comparison_data.append({
            "topic_id": topic,
            "extratrees_ap": float(et_ap[i]),
            "tar_ap": float(tar_ap[i]),
            "difference": float(et_ap[i] - tar_ap[i]),
            "winner": "ExtraTrees" if et_ap[i] > tar_ap[i] else "TAR" if tar_ap[i] > et_ap[i] else "Tie"
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("difference", ascending=False)

    print("\n" + "-" * 70)
    print("Per-Topic Breakdown:")
    print("-" * 70)
    print(comparison_df.to_string(index=False))

    # Save per-topic comparison
    comparison_path = OUTPUT_DIR / "clef_tar_per_topic_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved: {comparison_path}")

    # Save bootstrap summary
    summary_data = []
    for method_name, ap_values in [("TAR-Augmented ExtraTrees", et_ap), ("TAR TF-IDF+LogReg", tar_ap)]:
        summary_data.append({
            "method": method_name,
            "mean_ap": float(np.mean(ap_values)),
            "std_ap": float(np.std(ap_values)),
            "min_ap": float(np.min(ap_values)),
            "max_ap": float(np.max(ap_values)),
            "n_topics": len(ap_values),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = OUTPUT_DIR / "clef_tar_paired_bootstrap_summary.csv"

    # Add bootstrap statistics
    bootstrap_stats = pd.DataFrame([{
        "comparison": "ExtraTrees vs TAR",
        "mean_difference": float(mean_diff),
        "ci_lower_95": float(ci_lower),
        "ci_upper_95": float(ci_upper),
        "p_value": float(p_value),
        "p_value_display": f"< {1.0/N_BOOTSTRAP:.4f}" if raw_p == 0 else f"{p_value:.4f}",
        "significance": sig_level,
        "win_rate": float(mean_win_rate),
        "win_rate_ci_lower": float(win_rate_ci_lower),
        "win_rate_ci_upper": float(win_rate_ci_upper),
        "n_topics": len(et_ap),
        "et_wins": et_win_count,
        "tar_wins": tar_win_count,
        "ties": ties,
    }])

    # Combine into single file
    with open(summary_path, "w") as f:
        f.write("Method Statistics\n")
        summary_df.to_csv(f, index=False)
        f.write("\nBootstrap Statistics\n")
        bootstrap_stats.to_csv(f, index=False)

    print(f"Saved: {summary_path}")

    # Save detailed bootstrap distribution
    bootstrap_dist = pd.DataFrame({
        "mean_diff": boot_diffs,
        "win_rate": boot_win_rates,
    })

    bootstrap_dist_path = OUTPUT_DIR / "clef_tar_paired_bootstrap_ci.csv"
    bootstrap_dist.to_csv(bootstrap_dist_path, index=False)
    print(f"Saved: {bootstrap_dist_path}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if p_value < 0.05 and et_win_count >= 14:
        print("\n✓ STRONG EVIDENCE: ExtraTrees significantly outperforms TAR")
        print("  - p-value < 0.05")
        print(f"  - ExtraTrees wins on {et_win_count}/{len(et_ap)} topics")
        print("  - External validity established across diverse topics")
        print("\nRecommendation: Update manuscript with these results.")
    elif p_value < 0.10 or et_win_count >= 12:
        print("\n~ MODERATE EVIDENCE: ExtraTrees shows consistent improvement")
        print(f"  - p-value = {p_value:.4f}")
        print(f"  - ExtraTrees wins on {et_win_count}/{len(et_ap)} topics")
        print("\nRecommendation: Report as promising trend, consider additional analysis.")
    else:
        print("\n✗ WEAK EVIDENCE: Results not conclusive")
        print(f"  - p-value = {p_value:.4f}")
        print(f"  - ExtraTrees wins on {et_win_count}/{len(et_ap)} topics")
        print("\nRecommendation: Consider other approaches (Option C or D).")

    print("\nDone.")


if __name__ == "__main__":
    main()
