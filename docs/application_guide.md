# SPECTER-Triage Application Guide

## Overview

SPECTER-Triage provides a web application for scoring biomedical papers for systematic review relevance. The application combines a **FastAPI** backend serving trained reranker models with a **Vue.js** frontend for interactive exploration.

## Architecture

```
application/
├── api/
│   ├── main.py              # FastAPI backend
│   └── requirements.txt     # Python dependencies
└── frontend/
    └── index.html           # Vue.js single-page application
```

## Prerequisites

- Python 3.10+
- The SPECTER-Triage project outputs (in `outputs/`)

## Quick Start

### 1. Install API dependencies

From the project root directory:
```bash
pip install -r application/api/requirements.txt
```

### 2. Start the server

From the project root directory:
```bash
uvicorn application.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the application

- **Web UI**: http://localhost:8000/app
- **API docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Features

### Score Paper

Enter a paper title and abstract to get a relevance score from any of the trained models:
- TAR-Augmented ExtraTrees (primary model)
- Random Forest
- Logistic Regression
- Naive Bayes
- Gradient Boosting
- HistGradient Boosting
- AdaBoost
- SVM (Linear)
- TAR TF-IDF + LogReg (baseline)
- SPECTER-hybrid (manual baseline)

### Ranking Overview

View the top-ranked papers for any model, with relevance labels shown for evaluation.

### Feature Importance

SHAP-based feature importance showing which retrieval signals contribute most to the ExtraTrees model's decisions.

### Leave-One-Feature-Out Ablation

Performance impact when each feature is removed, proving the necessity of the multi-feature architecture.

### Statistics

Bootstrap confidence intervals and paired p-values for all core comparisons.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/models` | List available models with metrics |
| GET | `/ranking?model=X&top_k=50` | Get ranked records |
| POST | `/score` | Score a single paper |
| GET | `/feature-importance` | SHAP feature importance |
| GET | `/ablation` | LOFO ablation results |
| GET | `/statistics` | Bootstrap CIs and p-values |
| GET | `/metrics` | Per-model evaluation metrics |
| GET | `/app` | Web UI |

## Example: Score a Paper via API

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Mutational signature analysis using deep learning",
    "abstract": "We present a deep learning framework for extracting mutational signatures from cancer genomes...",
    "model": "learned_extratrees"
  }'
```

Response:

```json
{
  "relevance_score": 0.7342,
  "model": "learned_extratrees",
  "model_display": "TAR-Augmented ExtraTrees",
  "rank_features": {
    "BM25": 0.0,
    "TF-IDF": 0.0,
    "MiniLM": 0.0,
    "SPECTER": 0.0,
    "SPECTER-RQ": 0.0,
    "SPECTER-Proposal": 0.0,
    "Keyword": 0.0,
    "PubMedBERT": 0.0,
    "SPECTER-Hybrid": 0.0,
    "TAR TF-IDF+LogReg": 0.0,
    "Score Mean": 0.0,
    "Score Std": 0.0,
    "Lexical Mean": 0.0,
    "Dense Mean": 0.0,
    "Lexical−Dense": 0.0,
    "Consensus@100": 0
  },
  "interpretation": "High relevance — likely include"
}
```

> **Note**: The `/score` endpoint currently returns scores based on model priors. Full feature computation from raw text requires the embedding models (SPECTER, MiniLM, PubMedBERT, MedCPT) to be downloaded and loaded. See the [Reproducibility Guide](reproducibility_guide.md) for instructions on setting up the full feature pipeline.

## Advanced Features

### Using Your Own Dataset

The application can be retrained on a custom systematic review dataset. This requires two steps: generating the ranking scores, then uploading them to the app.

#### Step 1: Generate ranking scores from your data

Your dataset must have:
- A CSV with `record_id`, `title`, `abstract`, and `is_relevant` (binary label: 1 = included, 0 = excluded)
- At minimum, compute BM25, TF-IDF, and keyword coverage scores against your review query

Run the SPECTER-Triage pipeline on your data:

```bash
# 1. Prepare your data as a CSV with columns: record_id, title, abstract, is_relevant
# 2. Create study inventory and labels
python src/01_create_study_inventory.py
python src/02_create_screening_labels.py

# 3. Build ranking dataset
python src/03_create_ranking_dataset.py

# 4. Compute baseline scores (BM25, TF-IDF, keyword)
python src/04_run_baseline_rankers.py

# 5. Compute embedding scores (requires GPU, ~1-2 hours)
python src/05_run_specter_ranker.py
python src/05b_run_additional_embedding_rankers.py

# 6. Compute the manual SPECTER-hybrid score
# Edit 05_run_specter_ranker.py to output the hybrid score, or compute manually:
# hybrid = 0.65 * rq_similarity + 0.10 * proposal_similarity + 0.25 * keyword_score
```

Your final CSV must contain at least these columns:

| Column | Type | Required |
|--------|------|----------|
| `record_id` | string | Yes |
| `title` | string | Yes |
| `is_relevant` | int (0/1) | Yes |
| `bm25_score` | float | Yes |
| `tfidf_score` | float | Yes |
| `keyword_score` | float | Yes |
| `minilm_score` | float | Recommended |
| `specter_score` | float | Recommended |
| `specter_rq_similarity` | float | Recommended |
| `specter_proposal_similarity` | float | Recommended |
| `specter_hybrid_score` | float | Recommended |
| `pubmedbert_score` | float | Optional |

#### Step 2: Upload to the application

```bash
# Place your CSV at: outputs/ranking_scores.csv
# Then restart the API — it will auto-detect and train on your data:

uvicorn application.api.main:app --reload --host 0.0.0.0 --port 8000
```

Or use the API directly:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/ranking_scores.csv"
```

The API will:
1. Validate required columns exist
2. Auto-detect feature columns
3. Train all 8 learned reranker models on your data
4. Replace the in-memory models
5. Return per-model metrics (AP, Rel@100, Rank@90%)

#### What changes in the app

After uploading your dataset:
- **Score Paper** scores against your trained models
- **Ranking** shows your records ranked by your models
- **Feature Importance** shows which signals matter for your review topic
- **Ablation** shows which features are critical for your specific review
- **Statistics** shows bootstrap CIs for your data (you may need to re-run `07_bootstrap_metric_confidence_intervals.py` first)

#### Minimal working example

If you have a small dataset (e.g., 200 records, 20 relevant) and only computed BM25 + keyword scores:

```csv
record_id,title,abstract,is_relevant,bm25_score,tfidf_score,keyword_score
1,"Mutational signature analysis using NMF","We apply non-negative matrix factorization...",1,0.85,0.72,0.60
2,"Cancer genomics overview","A review of cancer genomic methods...",0,0.30,0.25,0.15
...
```

The app will still train models — they'll just be less accurate with fewer features. At minimum, BM25 + TF-IDF + keyword gives you a working baseline.

### Active Learning Simulation

The application includes a retrospective active-learning simulation that estimates how reviewer feedback could improve prioritisation during screening.

#### How it works

1. Start with an initial ranking (e.g., SPECTER-hybrid or BM25)
2. Screen the top N records (batch size: 25, 50, or 100)
3. Reveal labels for those records
4. Retrain the ExtraTrees model on the revealed labels
5. Re-rank remaining unscreened records
6. Repeat until 95% recall is reached or a stopping criterion triggers

#### Calibrated stopping rule

The simulation includes a calibrated stopping criterion: it halts when the recall improvement per batch falls below 1%, indicating diminishing returns. This helps reviewers decide when to stop screening.

On the primary dataset (2,231 records, 120 relevant):
- **Batch size 25**: stopped at round 13 (325 records, 14.6% screened, 90% recall)
- **Batch size 50**: completed at round 8 (400 records, 17.9% screened, 95% recall)
- **Batch size 100**: completed at round 4 (400 records, 17.9% screened, 95% recall)

#### Running the simulation

```bash
# Primary dataset
python src/05e_simulate_active_learning_triage.py

# CLEF-TAR external benchmark
python src/05f_active_learning_clef_tar.py
```

Outputs:
- `outputs/active_learning_simulation.csv` — per-round recall curves
- `outputs/figures/active_learning_recall_curve.png` — recall vs screening effort plot
- `outputs/figures/screening_burden_reduction.png` — screening burden comparison

#### Interpreting the results

| Metric | Meaning |
|--------|---------|
| Recall@50% = 100 records | Need to screen 100 records to find 50% of relevant studies |
| Recall@90% = 300 records | Need to screen 300 records to find 90% of relevant studies |
| 95% recall at 17.9% screened | 95% of relevant studies found after reviewing only 17.9% of the collection |
| Stopping at round 13 | Model signal suggests diminishing returns — remaining records unlikely to be relevant |

#### Using with your own data

After generating `ranking_scores.csv` for your dataset (see above), run:

```bash
# Edit 05e to point to your CSV, or pass via argument
python src/05e_simulate_active_learning_triage.py
```

The simulation will use your initial ranking and retrain on your labels. This gives you an estimate of how much screening effort active learning could save for your specific review.

## Notes

- The API loads all trained models into memory on startup (~30 seconds)
- Models are retrained from the existing ranking scores CSV on each startup
- The primary model (TAR-Augmented ExtraTrees) uses augmented features: rank percentiles, agreement features, and lexical-dense differences
- The Vue.js frontend communicates with the API via CORS-enabled HTTP requests
- For production deployment, consider using a process manager (e.g., gunicorn with uvicorn workers)
- Custom datasets require pre-computed feature scores (BM25, TF-IDF, embeddings) — the app does not compute embeddings from raw text on-the-fly
- Active learning results are retrospective simulations, not evidence of prospective deployment performance
