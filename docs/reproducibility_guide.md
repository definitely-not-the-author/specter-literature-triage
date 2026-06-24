# SPECTER-Triage Reproducibility Guide & Supplementary Materials

## Overview

This document provides:
1. **Reproducibility instructions** — how to reproduce all results from the manuscript
2. **Supplementary materials** — additional figures, tables, and analyses that support the manuscript

For the BIBM extension, model names are staged explicitly:

- **Standalone retrieval baselines**: BM25, TF-IDF query similarity, MiniLM, SPECTER, MedCPT, and related zero-label scores.
- **SPECTER-Triage Learned ExtraTrees**: non-TAR learned reranking over retrieval-score features.
- **TAR TF-IDF+LogReg**: supervised topic-specific TF-IDF logistic-regression TAR baseline.
- **TAR-Augmented ExtraTrees**: supervised learned reranking that includes TAR-derived scores together with semantic and lexical retrieval signals.

The CLEF-TAR `p = 0.0001` result applies to non-TAR SPECTER-Triage Learned ExtraTrees versus standalone retrieval baselines. The CLEF-TAR `p = 0.497` result applies to the harder supervised comparison between TAR-Augmented ExtraTrees and TAR TF-IDF+LogReg.

---

# Part 1: Reproducibility Instructions

## Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd specter-literature-triage

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Required Dependencies

Key packages (see `requirements.txt` for full list):
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `sentence-transformers>=2.2.0`
- `transformers>=4.30.0`
- `torch>=2.0.0`
- `shap>=0.42.0`
- `matplotlib>=3.7.0`
- `rank-bm25>=0.2.2`

## Data

The primary dataset is derived from a Covidence systematic review workflow on computational methods for mutational signature analysis. After excluding ambiguous audit records:
- **2,231 records** (title + abstract)
- **120 relevant** (included studies)
- **Prevalence**: 5.38%

The CLEF eHealth TAR benchmark contains:
- **149,671 records** across 20 Cochrane systematic reviews
- **2,515 relevant** records

## Pipeline Execution

Run scripts in order from `src/`:

```bash
# Step 1-3: Data preparation
python src/01_create_study_inventory.py
python src/02_create_screening_labels.py
python src/03_create_ranking_dataset.py

# Step 4: Baseline rankers (BM25, TF-IDF, keyword)
python src/04_run_baseline_rankers.py

# Step 5a-e: Embedding-based rankers
python src/05_run_specter_ranker.py
python src/05b_run_additional_embedding_rankers.py
cd outputs
python merge.py
cd ../
cp outputs/rankings/ranking_scores.csv outputs
python src/05c_train_learned_hybrid_reranker.py \
  --input outputs/ranking_scores.csv \
  --output outputs/ranking_scores_with_learned_reranker.csv \
  --metrics-output outputs/learned_reranker_metrics.csv \
  --coefficients-output outputs/learned_logistic_coefficients.csv
python src/05d_run_medcpt_baseline.py

# Step 5e: Active learning simulation (primary dataset only)
python src/05e_simulate_active_learning_triage.py

# Step 5g: TAR baseline
python src/05g_run_tfidf_logreg_tar_baseline.py

# Step 6: Main evaluation
python src/06_evaluate_rankings.py

# Step 6b-d: CLEF-TAR benchmarks (clones TAR repo, computes all CLEF scores)
python src/06b_public_benchmark_clef_tar.py
python src/06c_public_benchmark_clef_tar_medcpt.py
python src/06d_build_clef_tar_ranking_scores_TAR_augmented.py # Generates large files (~190 MB each)

# Note: src/06d_build_clef_tar_ranking_scores.py is retained for the
# non-augmented CLEF build. The BIBM v4 CLEF claims that include
# TAR-Augmented ExtraTrees require the suffixed TAR_augmented script above.

# Step 5f: CLEF-TAR active learning (requires 06b to have cloned TAR repo)
python src/05f_active_learning_clef_tar.py

# Step 7: Bootstrap CIs
python src/07_bootstrap_metric_confidence_intervals.py

# Step 7b: CLEF-TAR paired bootstrap (TAR-Augmented ExtraTrees vs TAR)
python src/07b_clef_tar_paired_bootstrap.py

# Step 7c: Result tables and figures
python src/07_generate_results_tables_figures.py

# Step 8: Error analysis
python src/08_error_analysis.py
python src/08b_error_analysis_clef_tar.py

# Step 8b-c: Ablation and interpretability
python src/08_run_ablation_analysis.py
python src/08b_lofo_ablation_extratrees.py
python src/08c_shap_feature_importance.py
python src/08d_error_case_studies.py

# Step 9: Screening efficiency
python src/09_generate_screening_efficiency_analysis.py

# Step 9c: Recovery depth (all methods)
python src/09c_recovery_depth_learned_rerankers.py

# Step 10: Additional figures
python src/09b_generate_extension_figures.py
python src/10_generate_additional_claim_figures.py
python src/10_generate_extension_comparison_table.py

# Step 11: Statistical rigor table
python src/11_generate_statistical_rigor_table.py
```

## Expected Outputs

After running the full pipeline, `outputs/` should contain:

```
outputs/
├── ranking_scores.csv
├── ranking_scores_with_learned_reranker.csv
├── learned_reranker_metrics.csv
├── learned_logistic_coefficients.csv
├── bootstrap_metric_ci.csv
├── bootstrap_metric_ci_summary.csv
├── active_learning_simulation.csv
├── shap_feature_importance.csv
├── figures/
│   ├── ranking_metrics_panel.png
│   ├── recall_at_k.png
│   ├── precision_at_k.png
│   ├── ndcg_at_k.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── lofo_ablation_ap.png
│   ├── lofo_ablation_rank90.png
│   ├── feature_importance_extratrees.png
│   ├── active_learning_recall_curve.png
│   └── ... (46 figure files total, counting PNG and PDF outputs)
├── tables/
│   ├── table_statistical_rigor.csv
│   ├── table_extension_comparison_main.csv
│   ├── table_recovery_depth_full.csv
│   ├── table_lofo_ablation.csv
│   └── ... (13 tables total)
├── metrics/
│   ├── ranking_metrics.csv
│   └── ablation_metrics.csv
├── public_benchmark/
│   ├── clef_tar_metrics.csv
│   ├── clef_tar_table_ranking_metrics.csv
│   ├── clef_tar_per_topic_comparison.csv
│   ├── clef_tar_paired_bootstrap_summary.csv
│   ├── clef_tar_paired_bootstrap_ci.csv
│   └── ... (19 files total, including generated large ranking files)
└── rankings/
    └── ... (10 ranking files)
```

**Note:** Two CLEF-TAR files are **not included in the repository** due to size (~190 MB each):
- `outputs/public_benchmark/clef_tar_ranking_scores.csv`
- `outputs/public_benchmark/clef_tar_ranking_scores_with_oof.csv`

These must be generated by running `python src/06d_build_clef_tar_ranking_scores_TAR_augmented.py` for the BIBM v4 TAR-augmented CLEF-TAR results.

## Source-of-Truth Result Files for BIBM v4

Use these files when checking manuscript claims:

| Claim Type | Source File |
|------------|-------------|
| Primary statistical AP, Rel@100, Rank@90% CIs and p-values | `outputs/tables/table_statistical_rigor.csv` |
| Primary direct AP, Rel@100, nDCG@100, recovery depth | `outputs/tables/table_extension_comparison_main.csv` |
| Primary screening-efficiency cutoffs | `outputs/tables/table_screening_efficiency_full.csv` |
| Primary recovery-depth table | `outputs/tables/table_recovery_depth_full.csv` |
| Current LOFO ablation | `outputs/tables/table_lofo_ablation.csv` |
| Current SHAP feature importance | `outputs/shap_feature_importance.csv` |
| CLEF-TAR per-topic mean performance | `outputs/public_benchmark/clef_tar_table_ranking_metrics.csv` |
| CLEF-TAR TAR-Augmented ExtraTrees vs TAR bootstrap | `outputs/public_benchmark/clef_tar_paired_bootstrap_summary.csv` |
| CLEF-TAR per-topic AP comparison | `outputs/public_benchmark/clef_tar_per_topic_comparison.csv` |

For CLEF-TAR Rank@90% claims, prefer `outputs/public_benchmark/clef_tar_table_ranking_metrics.csv` over the compact cross-dataset comparison table, because the dedicated CLEF table is computed as the per-topic benchmark summary used in the manuscript.

---

# Part 2: Supplementary Materials

## Table S1. Primary Dataset Direct Performance (Selected Methods)

Direct ranking metrics from `outputs/tables/table_extension_comparison_main.csv`. These values do not include bootstrap confidence intervals; use Table I in the manuscript or `table_statistical_rigor.csv` for CI and p-value claims.

| Method | AP | P@100 | R@100 | nDCG@100 | Rel@100 | Rank@50% | Rank@75% | Rank@90% |
|--------|-----|-------|-------|----------|---------|----------|----------|----------|
| TF-IDF | 0.381 | 0.52 | 0.433 | 0.513 | 52 | 125 | 422 | 1002 |
| BM25 | 0.372 | 0.49 | 0.408 | 0.480 | 49 | 152 | 410 | 970 |
| PubMedBERT | 0.220 | 0.27 | 0.225 | 0.244 | 27 | 228 | 499 | 797 |
| MiniLM | 0.539 | 0.55 | 0.458 | 0.597 | 55 | 110 | 205 | 392 |
| SPECTER | 0.462 | 0.48 | 0.400 | 0.487 | 48 | 119 | 218 | 325 |
| SPECTER-hybrid | 0.546 | 0.60 | 0.500 | 0.603 | 60 | 99 | 187 | 289 |
| TAR TF-IDF+LogReg | 0.630 | 0.62 | 0.517 | 0.680 | 62 | 97 | 175 | 315 |
| Learned NB | 0.456 | 0.47 | 0.392 | 0.514 | 47 | 129 | 185 | 289 |
| Learned RF | 0.641 | 0.63 | 0.525 | 0.656 | 63 | 87 | 179 | 262 |
| Learned Logistic | 0.674 | 0.61 | 0.508 | 0.641 | 61 | 99 | 153 | 225 |
| **TAR-Augmented ExtraTrees** | **0.684** | **0.67** | **0.558** | **0.723** | **67** | **84** | **166** | **245** |

## Table S2. Screening Efficiency at Multiple Cutoffs

| Method | Rel@10 | Rel@25 | Rel@50 | Rel@100 | Rel@200 | Recall@100 | Enrichment@100 |
|--------|--------|--------|--------|---------|---------|------------|----------------|
| Keyword | 1 | 9 | 14 | 27 | 50 | 0.225 | 5.02 |
| TF-IDF | 7 | 13 | 26 | 52 | 72 | 0.433 | 9.67 |
| BM25 | 5 | 14 | 27 | 49 | 72 | 0.408 | 9.11 |
| PubMedBERT | 1 | 9 | 14 | 27 | 51 | 0.225 | 5.02 |
| MiniLM | 8 | 17 | 37 | 55 | 88 | 0.458 | 10.23 |
| SPECTER | 7 | 13 | 25 | 48 | 85 | 0.400 | 8.92 |
| SPECTER-hybrid | 5 | 15 | 32 | 60 | 94 | 0.500 | 11.16 |
| TAR TF-IDF+LogReg | 9 | 21 | 38 | 62 | 94 | 0.517 | 11.53 |
| TAR-Augmented ExtraTrees | 9 | 23 | 37 | 67 | 98 | 0.558 | 12.46 |

## Table S3. Full Recovery Depth (All 16 Methods)

| Method | Rank@25% | Rank@50% | Rank@75% | Rank@90% | Screen% @90% |
|--------|----------|----------|----------|----------|--------------|
| Keyword | 120 | 270 | 593 | 933 | 41.8% |
| TF-IDF | 59 | 125 | 422 | 1002 | 44.9% |
| BM25 | 55 | 152 | 410 | 970 | 43.5% |
| PubMedBERT | 110 | 228 | 499 | 797 | 35.7% |
| MiniLM | 40 | 110 | 205 | 392 | 17.6% |
| SPECTER | 57 | 119 | 218 | 325 | 14.6% |
| SPECTER-hybrid | 47 | 99 | 187 | 289 | 13.0% |
| TAR TF-IDF+LogReg | 39 | 97 | 175 | 315 | 14.1% |
| Learned Logistic | 35 | 99 | 153 | 225 | 10.1% |
| Learned NB | 56 | 129 | 185 | 289 | 13.0% |
| Learned RF | 38 | 87 | 179 | 262 | 11.7% |
| **TAR-Augmented ExtraTrees** | **38** | **84** | **166** | **245** | **11.0%** |
| Learned GB | 33 | 95 | 170 | 267 | 12.0% |
| Learned HGB | 38 | 97 | 175 | 242 | 10.8% |
| Learned AdaBoost | 49 | 101 | 172 | 272 | 12.2% |
| Learned SVM-Linear | 35 | 90 | 158 | 241 | 10.8% |

## Table S4. CLEF-TAR Per-Topic Performance (Mean ± Std across 20 Cochrane Reviews)

| Method | Mean AP | Std AP | Mean Rel@100 | Mean Rank@90% |
|--------|---------|--------|--------------|---------------|
| TAR TF-IDF+LogReg | 0.460 | 0.219 | 43.9 | 1187 |
| TAR-Augmented ExtraTrees | 0.460 | 0.217 | 42.9 | 1233 |
| SPECTER-Triage Learned ExtraTrees | 0.323 | 0.187 | 32.4 | 1575 |
| MiniLM | 0.240 | 0.165 | 25.0 | 1224 |
| SPECTER | 0.228 | 0.147 | 24.6 | 1389 |
| TF-IDF | 0.176 | 0.152 | 17.7 | 1769 |
| MedCPT | 0.159 | 0.125 | 16.4 | 1901 |
| BM25 | 0.144 | 0.115 | 16.1 | 3419 |

Interpretation: SPECTER-Triage Learned ExtraTrees is the non-TAR learned reranker and should be compared against standalone retrieval baselines. TAR-Augmented ExtraTrees should be compared against the supervised TAR TF-IDF+LogReg baseline.

## Table S4b. CLEF-TAR Paired Bootstrap: SPECTER-Triage Learned ExtraTrees vs Standalone Retrieval Baselines

Paired bootstrap over the 20 CLEF-TAR topics, using per-topic AP values from `outputs/public_benchmark/clef_tar_learned_reranker_metrics.csv`. These comparisons support the external benchmark claim for non-TAR learned semantic-lexical reranking.

| Comparator | Mean AP Difference | 95% CI | p-value | Topic Wins |
|------------|-------------------:|--------|--------:|-----------:|
| BM25 | +0.1790 | [+0.1218, +0.2423] | 0.0001 | 19/20 |
| TF-IDF retrieval | +0.1479 | [+0.0874, +0.2147] | 0.0001 | 18/20 |
| MiniLM | +0.0832 | [+0.0430, +0.1252] | 0.0001 | 17/20 |
| SPECTER | +0.0951 | [+0.0556, +0.1404] | 0.0001 | 18/20 |
| MedCPT | +0.1649 | [+0.1039, +0.2291] | 0.0001 | 18/20 |

## Table S5. CLEF-TAR Error Analysis Summary

| Metric | Value |
|--------|-------|
| Total topics | 20 |
| Total records | 149,671 |
| Total relevant | 2,515 |
| Gained relevant (TAR-Augmented ET vs MiniLM baseline) | 343 |
| Lost relevant | 112 |
| Net gain | +231 |
| False positives in top 50 | 468 |
| False negatives beyond rank 100 | 1,658 |

## Table S6. Active Learning Simulation Results

| Batch Size | Rounds | Final Screened | Final Recall | Recall@50% | Recall@75% | Recall@90% |
|------------|--------|----------------|--------------|------------|------------|------------|
| 25 | 13 (early stop) | 325 | 90.0% | 100 | 200 | 300 |
| 50 | 8 | 400 | 95.0% | 100 | 200 | 300 |
| 100 | 4 | 400 | 95.0% | 100 | 200 | 300 |

The calibrated stopping criterion (recall improvement < 1% per batch) triggered early stop for batch_size=25 at round 13, demonstrating diminishing returns after ~14.6% of the collection is screened.

## Table S7. Leave-One-Feature-Out Ablation for TAR-Augmented ExtraTrees

Current LOFO output from `outputs/tables/table_lofo_ablation.csv`. Baseline (all features): AP = 0.680, Rel@100 = 69, Rank@90% = 247. Positive ΔAP means AP decreased when the feature was removed, so the feature helped AP under this retraining protocol. Positive ΔRank@90% means more records were required to reach 90% recall after removal.

| Feature Removed | AP After Removal | ΔAP | Rel@100 | ΔRel@100 | Rank@90% | ΔRank@90% |
|-----------------|-----------------:|----:|--------:|----------:|---------:|-----------:|
| TAR top-100 flag | 0.663 | +0.017 | 68 | +1 | 265 | +18 |
| BM25 top-100 flag | 0.671 | +0.009 | 68 | +1 | 248 | +1 |
| TF-IDF top-100 flag | 0.672 | +0.008 | 68 | +1 | 254 | +7 |
| Keyword top-100 flag | 0.672 | +0.008 | 68 | +1 | 247 | 0 |
| Raw TAR TF-IDF+LogReg score | 0.673 | +0.007 | 68 | +1 | 272 | +25 |
| PubMedBERT top-100 flag | 0.673 | +0.007 | 67 | +2 | 245 | −2 |
| Keyword score | 0.674 | +0.006 | 69 | 0 | 246 | −1 |
| TAR rank percentile | 0.676 | +0.004 | 67 | +2 | 260 | +13 |

The current LOFO table uses an augmented feature set with TAR-derived, rank-percentile, top-100, lexical, dense, and aggregate score features. It should not be compared directly with older 9-feature LOFO tables.

## Table S8. Error Case Studies — Failure Mode Classification

| Failure Mode | Count | Description |
|--------------|-------|-------------|
| Scope drift | 11 | Simulation, visualization, or benchmarking tools related to but not directly about signature extraction/analysis methodology |
| Topic tangential | 11 | Cancer genomics/driver mutation papers mentioning signatures peripherally |
| Unclassified | 9 | Papers not matching predefined categories |
| Methodology peripheral | 7 | General bioinformatics pipelines including signature analysis as one component |
| Terminology overlap | 4 | Papers using similar NMF/signature terminology for different applications |

## Table S9. CLEF-TAR Per-Topic Comparison: TAR-Augmented ExtraTrees vs TAR TF-IDF+LogReg

Paired bootstrap analysis across 20 CLEF-TAR topics. TAR TF-IDF+LogReg wins on 11/20 topics, TAR-Augmented ExtraTrees wins on 8/20, 1 tie (p = 0.497, not significant).

| Topic ID | TAR-Augmented ET AP | TAR AP | Difference | Winner |
|----------|---------------|--------|------------|--------|
| CD008643 | 0.170 | 0.089 | +0.082 | TAR-Augmented ET |
| CD009593 | 0.725 | 0.665 | +0.060 | TAR-Augmented ET |
| CD010409 | 0.759 | 0.701 | +0.058 | TAR-Augmented ET |
| CD007394 | 0.644 | 0.617 | +0.027 | TAR-Augmented ET |
| CD008686 | 0.032 | 0.011 | +0.021 | TAR-Augmented ET |
| CD009323 | 0.553 | 0.535 | +0.018 | TAR-Augmented ET |
| CD009591 | 0.672 | 0.667 | +0.004 | TAR-Augmented ET |
| CD011134 | 0.604 | 0.602 | +0.002 | TAR-Augmented ET |
| CD011549 | 0.000 | 0.000 | 0.000 | Tie |
| CD008054 | 0.457 | 0.458 | −0.002 | TAR |
| CD011548 | 0.369 | 0.373 | −0.004 | TAR |
| CD007427 | 0.437 | 0.443 | −0.005 | TAR |
| CD011975 | 0.398 | 0.411 | −0.012 | TAR |
| CD009944 | 0.601 | 0.620 | −0.020 | TAR |
| CD011984 | 0.393 | 0.424 | −0.031 | TAR |
| CD009020 | 0.481 | 0.512 | −0.032 | TAR |
| CD010771 | 0.657 | 0.689 | −0.032 | TAR |
| CD010632 | 0.431 | 0.470 | −0.040 | TAR |
| CD010438 | 0.225 | 0.266 | −0.041 | TAR |
| CD008691 | 0.600 | 0.647 | −0.048 | TAR |

## Table S10. Paired Bootstrap Statistics: TAR-Augmented ExtraTrees vs TAR TF-IDF+LogReg across CLEF-TAR Topics

| Statistic | Value |
|-----------|-------|
| TAR-Augmented ExtraTrees mean AP | 0.460 (SD = 0.212) |
| TAR mean AP | 0.460 (SD = 0.213) |
| Mean difference (ET − TAR) | +0.0003 |
| 95% CI | [−0.014, +0.017] |
| p-value | 0.497 |
| Win rate | 40% (8/20 topics) |
| Win rate 95% CI | [20%, 60%] |
| Bootstrap iterations | 10,000 |

---

# Part 3: Supplementary Figures

All figures are in `outputs/figures/` (600 DPI PNG), rendered inline below.

<table>
<tr>
<td align="center" width="33%"><b>S1. Precision@k</b><br><img src="../outputs/figures/precision_at_k.png" width="100%"></td>
<td align="center" width="33%"><b>S2. Recall@k</b><br><img src="../outputs/figures/recall_at_k.png" width="100%"></td>
<td align="center" width="33%"><b>S3. nDCG@k</b><br><img src="../outputs/figures/ndcg_at_k.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>S4. Ranking Metrics Panel</b><br><img src="../outputs/figures/ranking_metrics_panel.png" width="100%"></td>
<td align="center"><b>S5. Recall Recovery Curve</b><br><img src="../outputs/figures/recall_recovery_curve.png" width="100%"></td>
<td align="center"><b>S6. Top-100 Composition</b><br><img src="../outputs/figures/top100_composition_by_method.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>S7. Hybrid Score Distribution</b><br><img src="../outputs/figures/hybrid_score_distribution_by_label.png" width="100%"></td>
<td align="center"><b>S8. Ablation: AP by Weight</b><br><img src="../outputs/figures/ablation_average_precision.png" width="100%"></td>
<td align="center"><b>S9. Feature Importance (Gini)</b><br><img src="../outputs/figures/feature_importance_extratrees.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>S10. SHAP Summary</b><br><img src="../outputs/figures/shap_summary.png" width="100%"></td>
<td align="center"><b>S11. SHAP Bar</b><br><img src="../outputs/figures/shap_bar.png" width="100%"></td>
<td align="center"><b>S12. LOFO Ablation: AP</b><br><img src="../outputs/figures/lofo_ablation_ap.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>S13. LOFO Ablation: Rank@90%</b><br><img src="../outputs/figures/lofo_ablation_rank90.png" width="100%"></td>
<td align="center"><b>S14. Active Learning Recall</b><br><img src="../outputs/figures/active_learning_recall_curve.png" width="100%"></td>
<td align="center"><b>S15. AL vs Static</b><br><img src="../outputs/figures/active_learning_vs_static_recall.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>S16. Screening Burden</b><br><img src="../outputs/figures/screening_burden_comparison.png" width="100%"></td>
<td align="center"><b>S17. Screening Reduction</b><br><img src="../outputs/figures/screening_burden_reduction.png" width="100%"></td>
<td align="center"><b>S18. Screening Fraction</b><br><img src="../outputs/figures/screening_fraction_comparison.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>S19. Recovery Depth</b><br><img src="../outputs/figures/recovery_depth_comparison.png" width="100%"></td>
<td align="center"><b>S20. CLEF-TAR Active Learning</b><br><img src="../outputs/figures/clef_tar_active_learning_recall_curve.png" width="100%"></td>
<td align="center"><b>S21. Cumulative Relevant Retrieval</b><br><img src="../outputs/figures/cumulative_relevant_retrieval.png" width="100%"></td>
</tr>
<tr>
<td align="center"><b>S22. Enrichment@k</b><br><img src="../outputs/figures/enrichment_at_k.png" width="100%"></td>
<td align="center"><b>S23. Recall Curve (All Methods)</b><br><img src="../outputs/figures/recall_curve_all_methods.png" width="100%"></td>
<td></td>
</tr>
</table>

---

# Part 4: Data Files Reference

## Primary Dataset

| File | Description |
|------|-------------|
| `outputs/ranking_scores.csv` | Base ranking scores (9 methods) |
| `outputs/ranking_scores_with_learned_reranker.csv` | Extended with learned reranker OOF scores |
| `outputs/ranking_scores_with_tar_baseline.csv` | Extended with TAR TF-IDF+LogReg scores |

## Model Outputs

| File | Description |
|------|-------------|
| `outputs/learned_reranker_metrics.csv` | Per-model metrics (13 variants) |
| `outputs/learned_logistic_coefficients.csv` | Logistic regression feature coefficients |
| `outputs/shap_feature_importance.csv` | SHAP values per feature |

## Statistical Analysis

| File | Description |
|------|-------------|
| `outputs/bootstrap_metric_ci.csv` | Full bootstrap CI results (2000 iterations) |
| `outputs/bootstrap_metric_ci_summary.csv` | Delta and p-value summary |

## Error Analysis

| File | Description |
|------|-------------|
| `outputs/error_analysis_summary.csv` | Summary statistics |
| `outputs/error_analysis_lost.csv` | Records lost by ET vs hybrid |
| `outputs/error_analysis_gained.csv` | Records gained by ET vs hybrid |
| `outputs/error_analysis_et_vs_tar_lost.csv` | Records lost by ET vs TAR |
| `outputs/error_analysis_et_vs_tar_gained.csv` | Records gained by ET vs TAR |
| `outputs/error_analysis_false_positives_top100.csv` | False positives in top 100 |
| `outputs/error_analysis_false_negatives.csv` | False negatives beyond Rank@90% |
| `outputs/tables/table_error_case_studies.csv` | Classified failure modes |

## CLEF-TAR Benchmark

| File | Description |
|------|-------------|
| `outputs/public_benchmark/clef_tar_metrics.csv` | Per-topic per-method metrics |
| `outputs/public_benchmark/clef_tar_learned_reranker_metrics.csv` | Per-topic metrics for standalone retrieval, Learned ExtraTrees, TAR, and TAR-Augmented ExtraTrees |
| `outputs/public_benchmark/clef_tar_table_ranking_metrics.csv` | Aggregated ranking metrics |
| `outputs/public_benchmark/clef_tar_error_analysis_summary.csv` | Error analysis summary |
| `outputs/public_benchmark/clef_tar_per_topic_comparison.csv` | Per-topic AP comparison (TAR-Augmented ExtraTrees vs TAR) |
| `outputs/public_benchmark/clef_tar_paired_bootstrap_summary.csv` | Paired bootstrap statistics |
| `outputs/public_benchmark/clef_tar_paired_bootstrap_ci.csv` | Bootstrap distribution (10,000 iterations) |

**Note:** The following CLEF-TAR ranking score files are **not included in the repository** due to their size (~190 MB each):
- `outputs/public_benchmark/clef_tar_ranking_scores.csv`
- `outputs/public_benchmark/clef_tar_ranking_scores_with_oof.csv`

To generate these files, run:
```bash
python src/06d_build_clef_tar_ranking_scores_TAR_augmented.py
```

Alternatively, if you need these files in the repository, use [Git LFS](https://git-lfs.github.com/) to track large files:
```bash
git lfs track "outputs/public_benchmark/clef_tar_ranking_scores*.csv"
git add .gitattributes
```

---

# Citation

If you use SPECTER-Triage in your research, please cite:

> definitely-not-the-author (2026). SPECTER-Triage: Multi-Objective Evaluation of Semantic-Lexical Reranking for Biomedical Systematic Review Triage.

---

# License

This project is released for academic and research purposes.
