from pathlib import Path
import re
import pandas as pd


RAW_DIR = Path("data/raw")
LABEL_DIR = Path("data/labels")
OUTPUT_LABELS = LABEL_DIR / "screening_labels.csv"
OUTPUT_AUDIT = LABEL_DIR / "label_audit_unassigned.csv"
OUTPUT_DUPLICATES = LABEL_DIR / "label_audit_duplicates.csv"


FILES = {
    "included": {
        "path": RAW_DIR / "included.csv",
        "screening_label": "include",
        "is_final_candidate": 1,
        "candidate_type": "full_extraction",
        "confirmed_by_supervisor": "yes",
    },
    "excluded": {
        "path": RAW_DIR / "excluded.csv",
        "screening_label": "exclude",
        "is_final_candidate": 0,
        "candidate_type": "excluded_after_screening",
        "confirmed_by_supervisor": "not_applicable",
    },
    "irrelevant": {
        "path": RAW_DIR / "irrelevant.csv",
        "screening_label": "irrelevant",
        "is_final_candidate": 0,
        "candidate_type": "irrelevant",
        "confirmed_by_supervisor": "not_applicable",
    },
}


def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalise_doi(value):
    value = clean_text(value).lower()
    value = value.replace("https://doi.org/", "")
    value = value.replace("http://doi.org/", "")
    value = value.replace("doi:", "")
    return value.strip()


def normalise_title(value):
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def standardise(df):
    out = pd.DataFrame()
    out["record_id"] = df["Covidence #"].map(clean_text)
    out["title"] = df["Title"].map(clean_text)
    out["authors"] = df["Authors"].map(clean_text)
    out["abstract"] = df["Abstract"].map(clean_text)
    out["year"] = df["Published Year"]
    out["journal"] = df["Journal"].map(clean_text)
    out["doi"] = df["DOI"].map(clean_text)
    out["ref"] = df["Ref"].map(clean_text)
    out["study"] = df["Study"].map(clean_text)
    out["notes"] = df["Notes"].map(clean_text)
    out["tags"] = df["Tags"].map(clean_text)

    out["doi_norm"] = out["doi"].map(normalise_doi)
    out["title_norm"] = out["title"].map(normalise_title)
    out["combined_text"] = (out["title"] + ". " + out["abstract"]).map(clean_text)

    return out


def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    master_path = RAW_DIR / "screening.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master screening file: {master_path}")

    master = standardise(pd.read_csv(master_path))
    master["source_master"] = "screening"

    labelled_parts = []

    for source_name, meta in FILES.items():
        path = meta["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = standardise(pd.read_csv(path))
        df["source_file"] = source_name
        df["screening_label"] = meta["screening_label"]
        df["is_final_candidate"] = meta["is_final_candidate"]
        df["candidate_type"] = meta["candidate_type"]
        df["confirmed_by_supervisor"] = meta["confirmed_by_supervisor"]
        df["manual_category"] = ""
        df["reason"] = ""

        labelled_parts.append(df)

    labels = pd.concat(labelled_parts, ignore_index=True)

    # Audit duplicates by record_id first.
    duplicate_record_ids = labels[
        labels["record_id"].duplicated(keep=False) & labels["record_id"].ne("")
    ].sort_values("record_id")

    # If record_id is missing, DOI/title checks help detect duplicates.
    duplicate_dois = labels[
        labels["doi_norm"].duplicated(keep=False) & labels["doi_norm"].ne("")
    ].sort_values("doi_norm")

    duplicate_titles = labels[
        labels["title_norm"].duplicated(keep=False) & labels["title_norm"].ne("")
    ].sort_values("title_norm")

    duplicate_audit = pd.concat(
        [
            duplicate_record_ids.assign(duplicate_check="record_id"),
            duplicate_dois.assign(duplicate_check="doi"),
            duplicate_titles.assign(duplicate_check="title"),
        ],
        ignore_index=True,
    ).drop_duplicates()

    # Merge labels onto master by record_id.
    merged = master.merge(
        labels[
            [
                "record_id",
                "screening_label",
                "is_final_candidate",
                "candidate_type",
                "confirmed_by_supervisor",
                "manual_category",
                "reason",
                "source_file",
            ]
        ],
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    unassigned = merged[merged["screening_label"].isna()].copy()

    # Fill unassigned records with explicit audit label.
    merged["screening_label"] = merged["screening_label"].fillna("unassigned")
    merged["is_final_candidate"] = merged["is_final_candidate"].fillna(0).astype(int)
    merged["candidate_type"] = merged["candidate_type"].fillna("unassigned")
    merged["confirmed_by_supervisor"] = merged["confirmed_by_supervisor"].fillna("unknown")
    merged["manual_category"] = merged["manual_category"].fillna("")
    merged["reason"] = merged["reason"].fillna("")
    merged["source_file"] = merged["source_file"].fillna("none")

    final_cols = [
        "record_id",
        "title",
        "authors",
        "abstract",
        "year",
        "journal",
        "doi",
        "ref",
        "study",
        "screening_label",
        "is_final_candidate",
        "candidate_type",
        "confirmed_by_supervisor",
        "manual_category",
        "reason",
        "notes",
        "tags",
        "source_file",
        "combined_text",
    ]

    merged[final_cols].to_csv(OUTPUT_LABELS, index=False)
    unassigned.to_csv(OUTPUT_AUDIT, index=False)
    duplicate_audit.to_csv(OUTPUT_DUPLICATES, index=False)

    print("Saved labels:", OUTPUT_LABELS)
    print("Saved unassigned audit:", OUTPUT_AUDIT)
    print("Saved duplicate audit:", OUTPUT_DUPLICATES)

    print("\nMaster rows:", len(master))
    print("Labelled rows before merge:", len(labels))
    print("Final labelled master rows:", len(merged))

    print("\nLabel counts:")
    print(merged["screening_label"].value_counts(dropna=False))

    print("\nCandidate type counts:")
    print(merged["candidate_type"].value_counts(dropna=False))

    print("\nUnassigned records:", len(unassigned))
    if len(unassigned) > 0:
        print(unassigned[["record_id", "title", "doi"]].head(10).to_string(index=False))

    print("\nPotential duplicate audit rows:", len(duplicate_audit))
    if len(duplicate_audit) > 0:
        print(
            duplicate_audit[
                ["duplicate_check", "record_id", "title", "doi", "source_file", "screening_label"]
            ].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()