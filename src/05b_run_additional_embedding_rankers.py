from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PATH = Path("data/processed/ranking_dataset.csv")
OUTPUT_DIR = Path("outputs/rankings")
EMBEDDING_DIR = Path("outputs/embeddings")


RESEARCH_QUERY = """
computational methods for mutational signature analysis in cancer genomics,
including mutation signature extraction, signature assignment, non-negative
matrix factorisation, machine learning, deep learning, graph neural networks,
Bayesian models, benchmarking, genomic mutation patterns, mutational processes,
HIV-associated cancer, and cancer genome analysis
"""


MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PUBMEDBERT_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def save_ranking(df, score_col, method_name):
    out = df.copy()
    out["method"] = method_name
    out["score"] = out[score_col]
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    columns = [
        "rank",
        "method",
        "record_id",
        "title",
        "doi",
        "screening_label",
        "is_relevant",
        "score",
        "abstract",
    ]

    out[columns].to_csv(OUTPUT_DIR / f"ranking_{method_name}.csv", index=False)
    print(f"Saved {method_name}: {OUTPUT_DIR / f'ranking_{method_name}.csv'}")


def run_minilm(df):
    print(f"\nRunning MiniLM baseline: {MINILM_MODEL_NAME}")

    model = SentenceTransformer(MINILM_MODEL_NAME)

    embedding_path = EMBEDDING_DIR / "minilm_document_embeddings.npy"

    if embedding_path.exists():
        print(f"Loading cached MiniLM embeddings: {embedding_path}")
        doc_embeddings = np.load(embedding_path)
    else:
        print("Encoding documents with MiniLM...")
        doc_embeddings = model.encode(
            df["combined_text"].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(embedding_path, doc_embeddings)
        print(f"Saved MiniLM embeddings: {embedding_path}")

    query_embedding = model.encode(
        [RESEARCH_QUERY],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = cosine_similarity(doc_embeddings, query_embedding).ravel()

    df_out = df.copy()
    df_out["minilm_score"] = scores

    save_ranking(df_out, "minilm_score", "minilm")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    summed_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return summed_embeddings / summed_mask


def encode_pubmedbert_texts(texts, tokenizer, model, device, batch_size=8, max_length=512):
    embeddings = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            encoded = {key: value.to(device) for key, value in encoded.items()}

            output = model(**encoded)
            pooled = mean_pooling(output, encoded["attention_mask"])

            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())

            print(f"Encoded PubMedBERT batch {start // batch_size + 1} / {int(np.ceil(len(texts) / batch_size))}")

    return np.vstack(embeddings)


def run_pubmedbert(df):
    print(f"\nRunning PubMedBERT baseline: {PUBMEDBERT_MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(PUBMEDBERT_MODEL_NAME).to(device)

    embedding_path = EMBEDDING_DIR / "pubmedbert_document_embeddings.npy"

    if embedding_path.exists():
        print(f"Loading cached PubMedBERT embeddings: {embedding_path}")
        doc_embeddings = np.load(embedding_path)
    else:
        print("Encoding documents with PubMedBERT...")
        doc_embeddings = encode_pubmedbert_texts(
            df["combined_text"].tolist(),
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=8,
            max_length=512,
        )
        np.save(embedding_path, doc_embeddings)
        print(f"Saved PubMedBERT embeddings: {embedding_path}")

    query_embedding = encode_pubmedbert_texts(
        [RESEARCH_QUERY],
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=1,
        max_length=512,
    )

    scores = cosine_similarity(doc_embeddings, query_embedding).ravel()

    df_out = df.copy()
    df_out["pubmedbert_score"] = scores

    save_ranking(df_out, "pubmedbert_score", "pubmedbert")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["combined_text"] = df["combined_text"].fillna("").astype(str)

    print(f"Loaded ranking dataset: {len(df)} rows")
    print(df["screening_label"].value_counts())

    run_minilm(df)
    run_pubmedbert(df)

    print("\nDone. Added MiniLM and PubMedBERT rankings.")


if __name__ == "__main__":
    main()