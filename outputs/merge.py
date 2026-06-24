import pandas as pd
from pathlib import Path

base_dir = Path("./rankings/")

files = {
    "keyword": base_dir / "ranking_keyword.csv",
    "tfidf": base_dir / "ranking_tfidf.csv",
    "bm25": base_dir / "ranking_bm25.csv",
    "minilm": base_dir / "ranking_minilm.csv",
    "pubmedbert": base_dir / "ranking_pubmedbert.csv",
    "specter": base_dir / "ranking_specter.csv",
    "specter_hybrid": base_dir / "ranking_specter_hybrid.csv",
}

merged = None

for method, path in files.items():
    df = pd.read_csv(path)

    keep_cols = [
        "record_id",
        "title",
        "doi",
        "screening_label",
        "is_relevant",
        "abstract",
        "score",
    ]

    extra_cols = [
        "specter_rq_similarity",
        "specter_proposal_similarity",
        "keyword_score",
    ]

    keep_cols = [c for c in keep_cols + extra_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = df.rename(columns={"score": f"{method}_score"})

    if merged is None:
        merged = df
    else:
        # Only merge method-specific score/features not already present
        merge_cols = ["record_id"] + [
            c for c in df.columns
            if c != "record_id" and c not in merged.columns
        ]

        # Always include the method score
        if f"{method}_score" not in merge_cols and f"{method}_score" in df.columns:
            merge_cols.append(f"{method}_score")

        merged = merged.merge(
            df[merge_cols],
            on="record_id",
            how="outer"
        )

output_path = base_dir / "ranking_scores.csv"
merged.to_csv(output_path, index=False)

print("Wrote:", output_path)
print("Shape:", merged.shape)
print("Columns:")
print(merged.columns.tolist())