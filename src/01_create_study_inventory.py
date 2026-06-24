from pathlib import Path
import re
import pandas as pd


RAW_PATH = Path("data/raw/covidence_raw.csv")
OUTPUT_PATH = Path("data/processed/study_inventory.csv")


def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    expected_columns = [
        "Title",
        "Authors",
        "Abstract",
        "Published Year",
        "Published Month",
        "Journal",
        "Volume",
        "Issue",
        "Pages",
        "Accession Number",
        "DOI",
        "Ref",
        "Covidence #",
        "Study",
        "Notes",
        "Tags",
    ]

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    inventory = pd.DataFrame()

    inventory["record_id"] = df["Covidence #"].map(clean_text)
    inventory["title"] = df["Title"].map(clean_text)
    inventory["authors"] = df["Authors"].map(clean_text)
    inventory["abstract"] = df["Abstract"].map(clean_text)
    inventory["year"] = df["Published Year"]
    inventory["month"] = df["Published Month"].map(clean_text)
    inventory["journal"] = df["Journal"].map(clean_text)
    inventory["volume"] = df["Volume"].map(clean_text)
    inventory["issue"] = df["Issue"].map(clean_text)
    inventory["pages"] = df["Pages"].map(clean_text)
    inventory["accession_number"] = df["Accession Number"].map(clean_text)
    inventory["doi"] = df["DOI"].map(clean_text)
    inventory["pubmed_id_or_ref"] = df["Ref"].map(clean_text)
    inventory["study_name"] = df["Study"].map(clean_text)
    inventory["notes"] = df["Notes"].map(clean_text)
    inventory["tags"] = df["Tags"].map(clean_text)

    inventory["combined_text"] = (
        inventory["title"] + ". " + inventory["abstract"]
    ).map(clean_text)

    inventory["has_abstract"] = inventory["abstract"].str.len() > 0
    inventory["has_doi"] = inventory["doi"].str.len() > 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(inventory)}")
    print(f"Records with abstracts: {inventory['has_abstract'].sum()}")
    print(f"Records with DOI: {inventory['has_doi'].sum()}")
    print(f"Unique record IDs: {inventory['record_id'].nunique()}")

    if inventory["record_id"].duplicated().any():
        print("WARNING: Duplicate record IDs found.")
        print(inventory[inventory["record_id"].duplicated(keep=False)][["record_id", "title"]])


if __name__ == "__main__":
    main()