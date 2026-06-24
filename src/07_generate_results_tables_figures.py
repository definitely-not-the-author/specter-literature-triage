from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


METRICS_PATH = Path("outputs/metrics/ranking_metrics.csv")
SCORES_PATH = Path("outputs/ranking_scores_with_learned_reranker.csv")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")
K_VALUES = [10, 25, 50, 100, 200]

METHOD_LABELS = {
    "keyword": "Keyword",
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "minilm": "MiniLM",
    "pubmedbert": "PubMedBERT",
    "specter": "SPECTER",
    "specter_hybrid": "SPECTER-hybrid",
    "tar_tfidf_logreg": "TAR TF-IDF+LogReg",
    "tar_augmented_extratrees": "TAR-Augmented ExtraTrees",
}

# Plot order: lexical baselines first, neural encoders, manual hybrid, TAR,
# then the flagship TAR-augmented reranker.
METHOD_ORDER = [
    "keyword",
    "tfidf",
    "bm25",
    "minilm",
    "pubmedbert",
    "specter",
    "specter_hybrid",
    "tar_tfidf_logreg",
    "tar_augmented_extratrees",
]

# Per-method visual encoding. Colour = Okabe-Ito colourblind-safe palette;
# distinct markers + line styles so the figure also survives greyscale print.
METHOD_STYLE = {
    "keyword":        {"color": "#999999", "marker": "o", "ls": ":"},
    "tfidf":          {"color": "#E69F00", "marker": "s", "ls": "--"},
    "bm25":           {"color": "#56B4E9", "marker": "^", "ls": "--"},
    "minilm":         {"color": "#009E73", "marker": "D", "ls": "-"},
    "pubmedbert":     {"color": "#F0E442", "marker": "v", "ls": ":"},
    "specter":        {"color": "#0072B2", "marker": "P", "ls": "-"},
    "specter_hybrid": {"color": "#D55E00", "marker": "X", "ls": "-"},
    "tar_tfidf_logreg": {"color": "#6D4C41", "marker": "*", "ls": "-."},
    "tar_augmented_extratrees": {"color": "#CC79A7", "marker": "h", "ls": "-"},
}

# Headline methods drawn with extra weight so they read first.
EMPHASIS = {"specter_hybrid", "tar_tfidf_logreg", "tar_augmented_extratrees"}

SCORE_METHODS = {
    "tar_tfidf_logreg": "tar_tfidf_logreg_score",
    "tar_augmented_extratrees": "learned_extratrees_specter_triage_oof_score",
}


def set_pub_style():
    """Global rcParams tuned for camera-ready figures."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "pdf.fonttype": 42,  # embed editable TrueType, not Type-3 (journal-safe)
        "ps.fonttype": 42,
    })


def ndcg_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    ideal = np.sort(y_true)[::-1]
    dcg = float(np.sum(y_sorted[:k] / np.log2(np.arange(2, k + 2))))
    idcg = float(np.sum(ideal[:k] / np.log2(np.arange(2, k + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def metrics_from_scores(method, y_true, scores):
    order = np.argsort(scores)[::-1]
    total_relevant = max(int(np.sum(y_true)), 1)
    row = {
        "method": method,
        "n_records": len(y_true),
        "total_relevant": int(np.sum(y_true)),
        "average_precision": average_precision_score(y_true, scores),
    }
    for k in K_VALUES:
        relevant_at_k = int(np.sum(y_true[order[:k]]))
        row[f"precision_at_{k}"] = relevant_at_k / k
        row[f"recall_at_{k}"] = relevant_at_k / total_relevant
        row[f"ndcg_at_{k}"] = ndcg_at_k(y_true, scores, k)
        row[f"relevant_found_at_{k}"] = relevant_at_k
    return row


def append_flagship_methods(metrics):
    """Add TAR and TAR-Augmented ExtraTrees rows when the learned score file exists."""
    if not SCORES_PATH.exists():
        return metrics

    score_df = pd.read_csv(SCORES_PATH)
    if "is_relevant" not in score_df.columns:
        return metrics

    y_true = score_df["is_relevant"].to_numpy(dtype=int)
    existing = set(metrics["method"].astype(str))
    rows = []
    for method, col in SCORE_METHODS.items():
        if method in existing or col not in score_df.columns:
            continue
        rows.append(metrics_from_scores(method, y_true, score_df[col].fillna(0).to_numpy()))

    if not rows:
        return metrics
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)


def save_main_table(metrics):
    table = metrics[
        [
            "method",
            "average_precision",
            "precision_at_25",
            "precision_at_50",
            "precision_at_100",
            "recall_at_100",
            "ndcg_at_100",
            "relevant_found_at_100",
        ]
    ].copy()

    table["method"] = table["method"].map(lambda m: METHOD_LABELS.get(m, m))

    numeric_cols = [
        "average_precision",
        "precision_at_25",
        "precision_at_50",
        "precision_at_100",
        "recall_at_100",
        "ndcg_at_100",
    ]

    for col in numeric_cols:
        table[col] = table[col].round(3)

    output_path = TABLE_DIR / "table_ranking_metrics.csv"
    table.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(table.to_string(index=False))

def _draw_metric(ax, by_method, metric_prefix, ylabel, title):
    """Draw one metric-vs-k axis using the shared per-method style."""
    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        row = by_method[method]
        style = METHOD_STYLE[method]
        emph = method in EMPHASIS

        y_values = [row[f"{metric_prefix}_{k}"] for k in K_VALUES]
        ax.plot(
            K_VALUES,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=2.6 if emph else 1.6,
            markersize=7 if emph else 5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=4 if emph else 3,
            alpha=1.0 if emph else 0.9,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlim(K_VALUES[0] * 0.9, K_VALUES[-1] * 1.1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Top-$k$ ranked records")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.margins(x=0.02)


def _save(fig, output_name):
    """Write both a vector PDF (for LaTeX) and a high-res PNG (for preview)."""
    stem = Path(output_name).stem
    for ext in ("pdf", "png"):
        path = FIGURE_DIR / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_panel(metrics):
    """Hero figure: Precision | Recall | NDCG side-by-side, one shared legend."""
    by_method = {row["method"]: row for _, row in metrics.iterrows()}

    panels = [
        ("precision_at", "Precision", "(a) Precision@$k$"),
        ("recall_at", "Recall", "(b) Recall@$k$"),
        ("ndcg_at", "NDCG", "(c) NDCG@$k$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, (prefix, ylabel, title) in zip(axes, panels):
        _draw_metric(ax, by_method, prefix, ylabel, title)
    # Titles already name each metric; the shared y-axis carries a generic label.
    axes[0].set_ylabel("Score")
    for ax in axes[1:]:
        ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.04),
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "ranking_metrics_panel")


def plot_metric(metrics, metric_prefix, ylabel, output_name):
    """Single-metric figure (kept for standalone use); same styling as panel."""
    by_method = {row["method"]: row for _, row in metrics.iterrows()}

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    _draw_metric(ax, by_method, metric_prefix, ylabel,
                 f"{ylabel} across screening cut-offs")
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    _save(fig, output_name)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    set_pub_style()

    metrics = pd.read_csv(METRICS_PATH)
    metrics = append_flagship_methods(metrics)

    save_main_table(metrics)

    # Primary publication figure: all three metrics in one panel.
    plot_panel(metrics)

    # Standalone single-metric figures (optional, same styling).
    plot_metric(
        metrics,
        metric_prefix="precision_at",
        ylabel="Precision",
        output_name="precision_at_k.png",
    )

    plot_metric(
        metrics,
        metric_prefix="recall_at",
        ylabel="Recall",
        output_name="recall_at_k.png",
    )

    plot_metric(
        metrics,
        metric_prefix="ndcg_at",
        ylabel="NDCG",
        output_name="ndcg_at_k.png",
    )


if __name__ == "__main__":
    main()
