from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/labels/screening_labels.csv")
OUTPUT_PATH = Path("data/processed/ranking_dataset.csv")
AUDIT_PATH = Path("data/processed/ranking_dataset_excluded_audit.csv")


EXCLUDE_FROM_EVALUATION = {"unassigned", "duplicate"}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "record_id",
        "title",
        "abstract",
        "doi",
        "screening_label",
        "combined_text",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    excluded_audit = df[df["screening_label"].isin(EXCLUDE_FROM_EVALUATION)].copy()
    usable = df[~df["screening_label"].isin(EXCLUDE_FROM_EVALUATION)].copy()

    usable["is_relevant"] = usable["screening_label"].map(
        {
            "include": 1,
            "exclude": 0,
            "irrelevant": 0,
        }
    )

    if usable["is_relevant"].isna().any():
        unknown = usable[usable["is_relevant"].isna()]["screening_label"].unique()
        raise ValueError(f"Unknown screening labels found: {unknown}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    usable.to_csv(OUTPUT_PATH, index=False)
    excluded_audit.to_csv(AUDIT_PATH, index=False)

    print(f"Saved ranking dataset: {OUTPUT_PATH}")
    print(f"Rows usable for evaluation: {len(usable)}")
    print(f"Saved excluded audit: {AUDIT_PATH}")
    print(f"Rows excluded from evaluation: {len(excluded_audit)}")

    print("\nScreening labels used for evaluation:")
    print(usable["screening_label"].value_counts())

    print("\nBinary relevance labels:")
    print(usable["is_relevant"].value_counts())

    print("\nExcluded from evaluation:")
    if len(excluded_audit) > 0:
        print(excluded_audit[["record_id", "title", "doi", "screening_label"]].to_string(index=False))


if __name__ == "__main__":
    main()