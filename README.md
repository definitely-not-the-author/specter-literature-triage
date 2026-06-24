# SPECTER-Triage

**SPECTER-Triage: Multi-Objective Evaluation of Semantic-Lexical Reranking for Biomedical Systematic Review Triage**

A learned semantic-lexical reranking framework that combines sparse lexical and dense semantic signals through cross-validated supervised reranking for biomedical systematic review screening.

## Key Finding

> The best average-precision system is not necessarily the best reviewer-workload model. TAR TF-IDF+LogReg achieves the highest AP (0.632), while Learned ExtraTrees achieves the strongest high-recall workload reduction (Rank@90% = 268 vs 338).

## Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SPECTER-Triage Framework                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT RECORDS (Title + Abstract)                                   │
│       │                                                             │
│       ├──► BM25, TF-IDF, Keyword Coverage  (Sparse Lexical)         │
│       ├──► SPECTER, MiniLM, MedCPT, PubMedBERT  (Dense Semantic)    │
│       ├──► SPECTER-RQ, SPECTER-Proposal  (Component Scores)         │
│       │                                                             │
│       ▼                                                             │
│  5-Fold Stratified CV → ExtraTrees Reranker (10 features)           │
│       │                                                             │
│       ▼                                                             │
│  Prioritised Screening Queue → Human Reviewer                       │
│  (system prioritises, does NOT autonomously exclude)                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
git clone <repo-url>
cd specter-literature-triage
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Replication Guide

Run all scripts from the `specter-literature-triage/` directory. See [docs/reproducibility_guide.md](docs/reproducibility_guide.md) for the complete pipeline.

```bash
# Step 1: Data preparation
python src/01_create_study_inventory.py
python src/02_create_screening_labels.py
python src/03_create_ranking_dataset.py

# Step 2: Baseline rankers
python src/04_run_baseline_rankers.py

# Step 3: Embedding rankers
python src/05_run_specter_ranker.py
python src/05b_run_additional_embedding_rankers.py
python src/05d_run_medcpt_baseline.py

# Step 3.1: Merge outputs
cd outputs/
python merge.py
cd ../

# Step 4: Learned reranker
python src/05c_train_learned_hybrid_reranker.py \
  --input outputs/rankings/ranking_scores.csv \
  --output outputs/ranking_scores_with_learned_reranker.csv \
  --metrics-output outputs/learned_reranker_metrics.csv \
  --coefficients-output outputs/learned_logistic_coefficients.csv

# Step 5: TAR baseline + evaluation
cp outputs/rankings/ranking_scores.csv outputs/
python src/05g_run_tfidf_logreg_tar_baseline.py
python src/06_evaluate_rankings.py

# Step 6: CLEF-TAR external benchmark
python src/06b_public_benchmark_clef_tar.py
python src/06c_public_benchmark_clef_tar_medcpt.py
python src/06d_build_clef_tar_ranking_scores.py

# Step 7: Active learning simulation
python src/05e_simulate_active_learning_triage.py
python src/05f_active_learning_clef_tar.py

# Step 8: Bootstrap CIs
python src/07_bootstrap_metric_confidence_intervals.py
python src/07b_clef_tar_paired_bootstrap.py

# Step 9: Error analysis
python src/08_error_analysis.py
python src/08b_error_analysis_clef_tar.py
python src/08d_error_case_studies.py

# Step 10: Ablation + interpretability
python src/08_run_ablation_analysis.py
python src/08b_lofo_ablation_extratrees.py
python src/08c_shap_feature_importance.py

# Step 11: Tables and figures
python src/07_generate_results_tables_figures.py
python src/09_generate_screening_efficiency_analysis.py
python src/09b_generate_extension_figures.py
python src/09c_recovery_depth_learned_rerankers.py
python src/10_generate_additional_claim_figures.py
python src/10_generate_extension_comparison_table.py
python src/11_generate_statistical_rigor_table.py
```

## Project Structure

```
specter-literature-triage/
├── src/                        # All pipeline scripts
│   ├── 01-03                    # Data preparation
│   ├── 04-05g                   # Rankers and learned reranker
│   ├── 06-06d                   # Evaluation and benchmarks
│   ├── 07-07b                   # Bootstrap CIs and figures
│   ├── 08-08d                   # Error analysis and ablation
│   ├── 09-09c                   # Screening efficiency and recovery depth
│   ├── 10-10b                   # Extension figures and tables
│   └── 11                       # Statistical rigor table
├── application/                 # Deployable web application
│   ├── api/                     # FastAPI backend
│   └── frontend/                # Vue.js frontend
├── docs/
│   ├── application_guide.md     # How to run the web app
│   └── reproducibility_guide.md # Full pipeline + supplementary materials
├── data/
│   ├── labels/                  # Screening labels
│   ├── processed/               # Ranking dataset
│   └── public_benchmark/        # CLEF-TAR data
├── outputs/
│   ├── *.csv                    # Main dataset outputs
│   ├── figures/                 # 34 figures (PNG + PDF)
│   ├── tables/                  # 13 comparison tables
│   ├── metrics/                 # Ranking metrics
│   ├── rankings/                # Per-method ranking CSVs
│   ├── embeddings/              # Cached embeddings (.npy)
│   └── public_benchmark/        # CLEF-TAR outputs
├── paper/
│   └── bibm_extension/          # Manuscript (tracked + clean versions)
└── requirements.txt
```

## Key Results

### Primary Dataset (2,231 records, 120 relevant)

| Method | AP [95% CI] | Rel@100 | Rank@90% | p vs Hybrid | p vs TAR |
|--------|-------------|---------|----------|-------------|----------|
| Manual SPECTER-hybrid | 0.552 [0.456, 0.640] | 60 | 306 | ref | 0.974 |
| TAR TF-IDF+LogReg | **0.632 [0.544, 0.711]** | 61 | 338 | 0.026* | ref |
| Learned ExtraTrees | 0.610 [0.514, 0.700] | **66** | **268** | 0.025* | 0.704 |

### CLEF-TAR External Benchmark (149K records, 2,515 relevant)

| Method | Mean AP | Std AP | Mean Rank@90% |
|--------|---------|--------|---------------|
| BM25 | 0.144 | 0.115 | 3,419 |
| TF-IDF | 0.176 | 0.152 | 1,769 |
| MiniLM | 0.240 | 0.165 | 1,224 |
| **Learned ExtraTrees** | **0.275** | **0.179** | **935** |

### Leave-One-Feature-Out Ablation

| Feature Removed | ΔAP | ΔRank@90% | Interpretation |
|-----------------|------|-----------|----------------|
| BM25 | +0.016 | +1 | Most critical lexical feature |
| Keyword | +0.011 | +2 | Domain terminology anchor |
| PubMedBERT | +0.010 | +10 | Modest AP, moderate Rank@90% |
| MiniLM | +0.007 | +13 | Dense semantic complement |
| SPECTER-Hybrid | −0.003 | −9 | Slightly noisy; redundant with components |

## Web Application

The project includes a deployable FastAPI + Vue.js application:

```bash
cd specter-literature-triage
pip install -r application/api/requirements.txt
uvicorn application.api.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000/app
```

Features:
- **Score Paper** — score any paper against trained models
- **Ranking** — view top-ranked records for any model
- **Feature Importance** — SHAP analysis of which signals matter
- **Ablation** — LOFO ablation results
- **Statistics** — bootstrap CIs and p-values

Supports custom datasets and active learning simulation. See [docs/application_guide.md](docs/application_guide.md).

## Output Files

### Main Dataset

| File | Description |
|------|-------------|
| `ranking_scores_with_learned_reranker.csv` | Full dataset with all model scores |
| `learned_reranker_metrics.csv` | Per-model metrics (13 variants) |
| `bootstrap_metric_ci.csv` | Bootstrap CIs (2000 iterations, vs hybrid + TAR) |
| `shap_feature_importance.csv` | SHAP values per feature |
| `active_learning_simulation.csv` | AL simulation with calibrated stopping |
| `error_analysis_*.csv` | Lost/gained records, false positives/negatives |

### Supplementary Tables

| File | Description |
|------|-------------|
| `table_statistical_rigor.csv` | CIs + p-values for all methods |
| `table_extension_comparison_main.csv` | 11-method comparison |
| `table_recovery_depth_full.csv` | Recovery depth (16 methods) |
| `table_lofo_ablation.csv` | LOFO ablation (9 features) |
| `table_error_case_studies.csv` | Classified failure modes |

### Figures (34 files, PNG + PDF)

Key figures: `shap_summary.png`, `shap_bar.png`, `lofo_ablation_ap.png`, `lofo_ablation_rank90.png`, `recovery_depth_comparison.png`, `active_learning_recall_curve.png`, `ranking_metrics_panel.png`

## Citation

```bibtex
@inproceedings{specter_triage,
  title={SPECTER-Triage: Multi-Objective Evaluation of Semantic-Lexical Reranking for Biomedical Systematic Review Triage},
  author={definitely-not-the-author},
  year={2026}
}
```
