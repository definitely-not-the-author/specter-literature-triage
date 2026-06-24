"""Seaborn variant of the ranking-metrics panel, for visual comparison only.

Produces outputs/figures/ranking_metrics_panel_seaborn.{pdf,png}.
Same data and method styling intent as 07_generate_results_tables_figures.py.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METRICS_PATH = Path("outputs/metrics/ranking_metrics.csv")
FIGURE_DIR = Path("outputs/figures")

METHOD_LABELS = {
    "keyword": "Keyword",
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "minilm": "MiniLM",
    "pubmedbert": "PubMedBERT",
    "specter": "SPECTER",
    "specter_hybrid": "SPECTER-hybrid",
}
METHOD_ORDER = list(METHOD_LABELS.values())
K_VALUES = [10, 25, 50, 100, 200]
METRICS = [("precision_at", "Precision"), ("recall_at", "Recall"), ("ndcg_at", "NDCG")]

# Okabe-Ito, same as the matplotlib version, mapped by display label.
PALETTE = {
    "Keyword": "#999999",
    "TF-IDF": "#E69F00",
    "BM25": "#56B4E9",
    "MiniLM": "#009E73",
    "PubMedBERT": "#F0E442",
    "SPECTER": "#0072B2",
    "SPECTER-hybrid": "#D55E00",
}
MARKERS = {"Keyword": "o", "TF-IDF": "s", "BM25": "^", "MiniLM": "D",
           "PubMedBERT": "v", "SPECTER": "P", "SPECTER-hybrid": "X"}


def to_long(metrics):
    rows = []
    for _, r in metrics.iterrows():
        label = METHOD_LABELS.get(r["method"], r["method"])
        for prefix, metric in METRICS:
            for k in K_VALUES:
                rows.append({"Method": label, "Metric": metric,
                             "k": k, "Score": r[f"{prefix}_{k}"]})
    return pd.DataFrame(rows)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    long = to_long(pd.read_csv(METRICS_PATH))

    g = sns.relplot(
        data=long,
        x="k", y="Score",
        hue="Method", style="Method",
        hue_order=METHOD_ORDER, style_order=METHOD_ORDER,
        palette=PALETTE, markers=MARKERS, dashes=False,
        col="Metric", col_order=[m for _, m in METRICS],
        kind="line", linewidth=2.2, markersize=9, markeredgecolor="white",
        height=4.0, aspect=0.95, facet_kws={"sharey": True},
    )
    g.set(xscale="log", xticks=K_VALUES, xlim=(9, 220), ylim=(0, 1))
    for ax in g.axes.flat:
        ax.set_xticklabels(K_VALUES)
        ax.set_xlabel("Top-$k$ ranked records")
    g.set_titles("{col_name}@$k$")
    g.axes.flat[0].set_ylabel("Score")
    sns.move_legend(g, "lower center", ncol=len(METHOD_ORDER),
                    bbox_to_anchor=(0.42, -0.08), title=None, frameon=False)

    for ext in ("pdf", "png"):
        path = FIGURE_DIR / f"ranking_metrics_panel_seaborn.{ext}"
        g.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(g.figure)


if __name__ == "__main__":
    main()
