import argparse
import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found: {candidates}\nAvailable columns: {list(df.columns)}")


def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    top = y_true[order][:k]
    return float(np.mean(top)) if len(top) else 0.0


def recall_at_k(y_true, scores, k):
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    top = y_true[order][:k]
    return float(np.sum(top) / total_pos)


def rank_at_recall(y_true, scores, target_recall=0.90):
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return np.nan

    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    cum_rel = np.cumsum(y_sorted)
    recall = cum_rel / total_pos

    hits = np.where(recall >= target_recall)[0]
    if len(hits) == 0:
        return np.nan
    return int(hits[0] + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/ranking_scores.csv",
        help="CSV with record_id, title, abstract, and labels if available."
    )
    parser.add_argument(
        "--text-input",
        default=None,
        help="Optional CSV with record_id, title, abstract, and labels if input does not contain text."
    )
    parser.add_argument(
        "--output",
        default="outputs/rankings/ranking_tfidf_logreg_tar.csv"
    )
    parser.add_argument(
        "--merged-output",
        default="outputs/ranking_scores_with_tar_baseline.csv"
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/tfidf_logreg_tar_metrics.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # If ranking_scores.csv does not contain text, merge from another dataset file.
    has_text = ("title" in df.columns) or ("abstract" in df.columns)
    if not has_text:
        if args.text_input is None:
            candidate_files = [
                "data/processed/ranking_dataset.csv",
                "data/processed/screening_dataset.csv",
                "outputs/ranking_dataset.csv",
            ]
            found = None
            for f in candidate_files:
                if os.path.exists(f):
                    found = f
                    break
            if found is None:
                raise FileNotFoundError(
                    "No title/abstract columns found in input, and no --text-input provided. "
                    "Provide the processed dataset CSV containing record_id, title, abstract, and labels."
                )
            args.text_input = found

        text_df = pd.read_csv(args.text_input)
        record_col_main = find_col(df, ["record_id", "id", "Record ID"])
        record_col_text = find_col(text_df, ["record_id", "id", "Record ID"])

        keep_cols = [record_col_text]
        for c in ["title", "abstract", "label", "is_relevant", "relevant", "included"]:
            if c in text_df.columns:
                keep_cols.append(c)

        df = df.merge(
            text_df[keep_cols],
            left_on=record_col_main,
            right_on=record_col_text,
            how="left",
            suffixes=("", "_text")
        )

    record_col = find_col(df, ["record_id", "id", "Record ID"])
    label_col = find_col(df, ["is_relevant", "relevant", "included", "label"])

    title = df["title"].fillna("") if "title" in df.columns else ""
    abstract = df["abstract"].fillna("") if "abstract" in df.columns else ""
    text = (title.astype(str) + " " + abstract.astype(str)).str.strip()

    y = df[label_col].astype(int).to_numpy()

    print(f"Records: {len(df)}")
    print(f"Relevant: {int(y.sum())}")
    print(f"Prevalence: {y.mean():.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = np.zeros(len(df), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(cv.split(text, y), start=1):
        print(f"Fold {fold}")

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=100_000,
                    sublinear_tf=True,
                )
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                )
            )
        ])

        model.fit(text.iloc[train_idx], y[train_idx])
        oof_scores[test_idx] = model.predict_proba(text.iloc[test_idx])[:, 1]

    ap = average_precision_score(y, oof_scores)
    p100 = precision_at_k(y, oof_scores, 100)
    r100 = recall_at_k(y, oof_scores, 100)
    rank90 = rank_at_recall(y, oof_scores, 0.90)

    metrics = pd.DataFrame([{
        "method": "tar_tfidf_logreg",
        "average_precision": ap,
        "precision_at_100": p100,
        "recall_at_100": r100,
        "relevant_at_100": int(round(r100 * y.sum())),
        "rank_at_90_recall": rank90,
    }])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)

    ranking = pd.DataFrame({
        record_col: df[record_col],
        "tar_tfidf_logreg_score": oof_scores,
        label_col: y,
    })

    ranking.to_csv(args.output, index=False)
    metrics.to_csv(args.metrics_output, index=False)

    merged = df.copy()
    merged["tar_tfidf_logreg_score"] = oof_scores
    merged.to_csv(args.merged_output, index=False)

    print("\nTAR-style TF-IDF + Logistic Regression baseline")
    print(metrics.to_string(index=False))
    print(f"\nSaved ranking: {args.output}")
    print(f"Saved merged scores: {args.merged_output}")
    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()