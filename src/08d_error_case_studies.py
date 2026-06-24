#!/usr/bin/env python3
"""
08d_error_case_studies.py

Purpose
-------
Generate narrative error case studies from the existing error analysis CSVs.
Adds interpretive columns (failure_mode, lesson) and outputs a formatted
table for the manuscript.

Usage
-----
python src/08d_error_case_studies.py

Outputs:
  outputs/tables/table_error_case_studies.csv
"""

from pathlib import Path
import pandas as pd


ET_VS_TAR_LOST = Path("outputs/error_analysis_et_vs_tar_lost.csv")
ET_VS_HYBRID_LOST = Path("outputs/error_analysis_lost.csv")
ET_VS_TAR_GAINED = Path("outputs/error_analysis_et_vs_tar_gained.csv")
OUTPUT_DIR = Path("outputs/tables")


FAILURE_MODES = {
    "NMF": "terminology_overlap",
    "non-negative matrix factorization": "terminology_overlap",
    "simulator": "scope_drift",
    "simulation": "scope_drift",
    "visualization": "scope_drift",
    "visualiz": "scope_drift",
    "pipeline": "methodology_peripheral",
    "workflow": "methodology_peripheral",
    "framework": "methodology_peripheral",
    "benchmark": "methodology_peripheral",
    "benchmarking": "methodology_peripheral",
    "driver": "topic_tangential",
    "prognost": "topic_tangential",
    "clinical": "topic_tangential",
    "landscape": "topic_tangential",
    "pan-cancer": "topic_tangential",
    "whole-genome": "topic_tangential",
    "precision oncology": "topic_tangential",
    "unknown primary": "topic_tangential",
    "immunotherapy": "topic_tangential",
}

LESSONS = {
    "terminology_overlap": "Papers using similar NMF/signature terminology but for different applications (e.g., deep unfolding for NMF) are hard to distinguish from actual signature methods.",
    "scope_drift": "Tools that simulate, visualise, or benchmark signature methods are related but not directly about signature extraction/analysis methodology.",
    "methodology_peripheral": "Papers describing general bioinformatics pipelines or frameworks that include signature analysis as one component are borderline relevant.",
    "topic_tangential": "Papers about cancer genomics, driver mutations, or clinical applications that mention signatures peripherally are ranked highly due to domain overlap.",
}


def classify_failure_mode(title, abstract):
    text = (title + " " + abstract).lower()
    for keyword, mode in FAILURE_MODES.items():
        if keyword.lower() in text:
            return mode
    return "unclassified"


def truncate_text(text, max_len=120):
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_studies = []

    if ET_VS_TAR_LOST.exists():
        df_lost = pd.read_csv(ET_VS_TAR_LOST)
        for _, row in df_lost.iterrows():
            mode = classify_failure_mode(str(row.get("title", "")), str(row.get("abstract", "")))
            case_studies.append({
                "record_id": row.get("record_id", ""),
                "title_short": truncate_text(str(row.get("title", "")), 80),
                "screening_label": row.get("screening_label", ""),
                "is_relevant": row.get("is_relevant", 0),
                "et_rank": row.get("et_rank", ""),
                "tar_rank": row.get("tar_rank", ""),
                "et_score": round(float(row.get("et_score", 0)), 3),
                "tar_score": round(float(row.get("tar_score", 0)), 3),
                "comparison": "ET vs TAR",
                "direction": "lost_by_ET",
                "failure_mode": mode,
                "lesson": LESSONS.get(mode, ""),
            })

    if ET_VS_HYBRID_LOST.exists():
        df_lost_h = pd.read_csv(ET_VS_HYBRID_LOST)
        for _, row in df_lost_h.head(10).iterrows():
            mode = classify_failure_mode(str(row.get("title", "")), str(row.get("abstract", "")))
            case_studies.append({
                "record_id": row.get("record_id", ""),
                "title_short": truncate_text(str(row.get("title", "")), 80),
                "screening_label": row.get("screening_label", ""),
                "is_relevant": row.get("is_relevant", 0),
                "et_rank": row.get("et_rank", ""),
                "hybrid_rank": row.get("hybrid_rank", ""),
                "et_score": round(float(row.get("et_score", 0)), 3),
                "hybrid_score": round(float(row.get("hybrid_score", 0)), 3),
                "comparison": "ET vs Hybrid",
                "direction": "lost_by_ET",
                "failure_mode": mode,
                "lesson": LESSONS.get(mode, ""),
            })

    if ET_VS_TAR_GAINED.exists():
        df_gained = pd.read_csv(ET_VS_TAR_GAINED)
        for _, row in df_gained.head(5).iterrows():
            mode = classify_failure_mode(str(row.get("title", "")), str(row.get("abstract", "")))
            case_studies.append({
                "record_id": row.get("record_id", ""),
                "title_short": truncate_text(str(row.get("title", "")), 80),
                "screening_label": row.get("screening_label", ""),
                "is_relevant": row.get("is_relevant", 0),
                "et_rank": row.get("et_rank", ""),
                "tar_rank": row.get("tar_rank", ""),
                "et_score": round(float(row.get("et_score", 0)), 3),
                "tar_score": round(float(row.get("tar_score", 0)), 3),
                "comparison": "ET vs TAR",
                "direction": "gained_by_ET",
                "failure_mode": mode,
                "lesson": "",
            })

    result_df = pd.DataFrame(case_studies)
    result_df.to_csv(OUTPUT_DIR / "table_error_case_studies.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'table_error_case_studies.csv'}")

    print("\n=== Failure Mode Distribution ===")
    mode_counts = result_df["failure_mode"].value_counts()
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}")

    print("\n=== Representative Case Studies (Lost by ET vs TAR) ===")
    tar_lost = result_df[(result_df["comparison"] == "ET vs TAR") & (result_df["direction"] == "lost_by_ET")]
    for _, row in tar_lost.head(5).iterrows():
        print(f"\n  {row['record_id']}: {row['title_short']}")
        print(f"    Label: {row['screening_label']}, ET rank: {row['et_rank']}, TAR rank: {row['tar_rank']}")
        print(f"    Failure mode: {row['failure_mode']}")
        if row['lesson']:
            print(f"    Lesson: {row['lesson'][:100]}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
