#!/usr/bin/env python3
"""
11_generate_statistical_rigor_table.py

Purpose
-------
Generate a statistical rigor table showing AP, Rel@100, Rank@90% with
95% CIs and p-values vs both manual SPECTER-hybrid and TAR TF-IDF+LogReg.

Usage
-----
python src/11_generate_statistical_rigor_table.py

Outputs:
  outputs/tables/table_statistical_rigor.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BOOTSTRAP_CI_PATH = Path("outputs/bootstrap_metric_ci.csv")
OUTPUT_DIR = Path("outputs/tables")

KEY_METRICS = ["ap", "relevant_at_100", "rank_at_90"]
METRIC_LABELS = {
    "ap": "AP",
    "relevant_at_100": "Rel@100",
    "rank_at_90": "Rank@90%",
}

METHOD_ORDER = [
    "manual_specter_hybrid",
    "tar_tfidf_logreg",
    "learned_extratrees",
    "learned_rf",
    "learned_nb",
    "learned_logistic",
    "learned_gb",
    "learned_hgb",
    "learned_adaboost",
    "learned_svm_linear",
]

METHOD_LABELS = {
    "manual_specter_hybrid": "Manual SPECTER-hybrid",
    "tar_tfidf_logreg": "TAR TF-IDF+LogReg",
    "learned_extratrees": "TAR-Augmented ExtraTrees",
    "learned_rf": "Learned RF",
    "learned_nb": "Learned NB",
    "learned_logistic": "Learned Logistic",
    "learned_gb": "Learned GB",
    "learned_hgb": "Learned HGB",
    "learned_adaboost": "Learned AdaBoost",
    "learned_svm_linear": "Learned SVM-Linear",
}


def format_ci(mean, lower, upper, decimals=3):
    return f"{mean:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def format_p(p):
    if p is None:
        return "-"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}**"
    if p < 0.05:
        return f"{p:.3f}*"
    return f"{p:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Generate statistical rigor table")
    parser.add_argument("--input", default=str(BOOTSTRAP_CI_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)

    rows = []
    for method in METHOD_ORDER:
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue

        row = {"Method": METHOD_LABELS.get(method, method)}

        for metric in KEY_METRICS:
            metric_df = method_df[method_df["metric"] == metric]
            if metric_df.empty:
                continue
            r = metric_df.iloc[0]
            label = METRIC_LABELS[metric]

            mean_val = r["mean"]
            ci_lower = r["ci_lower_95"]
            ci_upper = r["ci_upper_95"]
            decimals = 0 if metric in ("relevant_at_100", "rank_at_90") else 3

            row[f"{label}"] = format_ci(mean_val, ci_lower, ci_upper, decimals)

            p_hybrid = r.get("p_value_vs_manual_specter_hybrid")
            p_tar = r.get("p_value_vs_tar_tfidf_logreg")

            if method != "manual_specter_hybrid":
                row[f"p vs Hybrid ({label})"] = format_p(p_hybrid)
            else:
                row[f"p vs Hybrid ({label})"] = "ref"

            if method != "tar_tfidf_logreg":
                row[f"p vs TAR ({label})"] = format_p(p_tar)
            else:
                row[f"p vs TAR ({label})"] = "ref"

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "table_statistical_rigor.csv", index=False)
    print(f"\nSaved: {output_dir / 'table_statistical_rigor.csv'}")

    print("\nFormatted table:")
    print(result_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
