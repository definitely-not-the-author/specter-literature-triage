from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


RANKINGS_DIR = Path("outputs/rankings")
METRICS_DIR = Path("outputs/metrics")
SCORES_PATH = Path("outputs/ranking_scores_with_learned_reranker.csv")
FIGURE_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")

HYBRID_RANKING = RANKINGS_DIR / "ranking_specter_hybrid.csv"
ABLATION_METRICS = METRICS_DIR / "ablation_metrics.csv"

METHOD_FILES = {
    "keyword": RANKINGS_DIR / "ranking_keyword.csv",
    "tfidf": RANKINGS_DIR / "ranking_tfidf.csv",
    "bm25": RANKINGS_DIR / "ranking_bm25.csv",
    "minilm": RANKINGS_DIR / "ranking_minilm.csv",
    "pubmedbert": RANKINGS_DIR / "ranking_pubmedbert.csv",
    "specter": RANKINGS_DIR / "ranking_specter.csv",
    "specter_hybrid": RANKINGS_DIR / "ranking_specter_hybrid.csv",
}

SCORE_METHODS = {
    "tar_tfidf_logreg": "tar_tfidf_logreg_score",
    "tar_augmented_extratrees": "learned_extratrees_specter_triage_oof_score",
}

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

# Screening-outcome encoding for the label-based figures. Lightness is ordered
# (dark include -> light irrelevant) so the categories also separate in greyscale.
LABEL_ORDER = ["include", "exclude", "irrelevant"]
LABEL_STYLE = {
    "include":    {"color": "#009E73", "label": "Include"},
    "exclude":    {"color": "#D55E00", "label": "Exclude"},
    "irrelevant": {"color": "#BFBFBF", "label": "Irrelevant"},
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


def _save(fig, output_name):
    """Write both a vector PDF (for LaTeX) and a high-res PNG (for preview)."""
    stem = Path(output_name).stem
    for ext in ("pdf", "png"):
        path = FIGURE_DIR / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)


def load_ranked_method(method):
    """Load a ranked dataframe for either file-based baselines or score columns."""
    if method in METHOD_FILES:
        return pd.read_csv(METHOD_FILES[method]).sort_values("rank").reset_index(drop=True)

    if method in SCORE_METHODS:
        if not SCORES_PATH.exists():
            raise FileNotFoundError(f"Missing score file: {SCORES_PATH}")
        df = pd.read_csv(SCORES_PATH)
        col = SCORE_METHODS[method]
        if col not in df.columns:
            raise ValueError(f"Missing score column for {method}: {col}")
        df = df.copy()
        df["score"] = df[col].fillna(0)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    raise KeyError(f"Unknown method: {method}")


def plot_hybrid_score_distribution():
    df = pd.read_csv(HYBRID_RANKING)

    if "score" not in df.columns or "screening_label" not in df.columns:
        raise ValueError("Hybrid ranking must contain score and screening_label columns.")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for label in LABEL_ORDER:
        subset = df[df["screening_label"] == label]
        if len(subset) == 0:
            continue
        style = LABEL_STYLE[label]
        ax.hist(
            subset["score"],
            bins=30,
            density=True,
            histtype="stepfilled",
            color=style["color"],
            edgecolor=style["color"],
            linewidth=1.2,
            alpha=0.45,
            label=f"{style['label']} ($n={len(subset)}$)",
        )

    ax.set_xlabel("SPECTER-hybrid score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of hybrid relevance scores by screening outcome",
                 loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    _save(fig, "hybrid_score_distribution_by_label")


def plot_top100_composition():
    rows = []

    for method in METHOD_ORDER:
        df = load_ranked_method(method).head(100)
        counts = df["screening_label"].value_counts().to_dict()

        row = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
        }
        for label in LABEL_ORDER:
            row[label] = counts.get(label, 0)
        rows.append(row)

    comp = pd.DataFrame(rows)

    bottom = np.zeros(len(comp))
    x = np.arange(len(comp))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    for label in LABEL_ORDER:
        values = comp[label].to_numpy()
        style = LABEL_STYLE[label]
        ax.bar(
            x, values, bottom=bottom,
            color=style["color"], edgecolor="white", linewidth=0.6,
            label=style["label"],
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(comp["method_label"], rotation=30, ha="right")
    ax.set_ylabel("Number of records in top 100")
    ax.set_title("Top-100 screening outcome composition by ranking method",
                 loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(
        ncol=len(LABEL_ORDER),
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "top100_composition_by_method")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLE_DIR / "top100_composition_by_method.csv"
    comp.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")


def plot_recall_recovery_curve():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for method in METHOD_ORDER:
        df = load_ranked_method(method)
        relevance = df["is_relevant"].astype(int).to_numpy()
        total_relevant = relevance.sum()

        cumulative_recall = np.cumsum(relevance) / total_relevant
        x = np.arange(1, len(cumulative_recall) + 1)

        style = METHOD_STYLE[method]
        emph = method in EMPHASIS
        ax.plot(
            x, cumulative_recall,
            color=style["color"],
            linestyle=style["ls"],
            marker=style["marker"],
            markevery=25,
            markersize=6 if emph else 4,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=2.6 if emph else 1.6,
            zorder=4 if emph else 3,
            alpha=1.0 if emph else 0.9,
            label=METHOD_LABELS.get(method, method),
        )

    for y in [0.25, 0.50, 0.75]:
        ax.axhline(y=y, linestyle=(0, (1, 3)), linewidth=0.8, color="#888888",
                   zorder=1)

    ax.set_xlabel("Number of records screened")
    ax.set_ylabel("Recall of final included studies")
    ax.set_title("Recall recovery by screening depth", loc="left")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    _save(fig, "recall_recovery_curve")


def plot_ablation_average_precision():
    if not ABLATION_METRICS.exists():
        print(f"Skipping ablation chart. Missing: {ABLATION_METRICS}")
        return

    df = pd.read_csv(ABLATION_METRICS)

    if "variant" not in df.columns or "average_precision" not in df.columns:
        raise ValueError("Ablation metrics must contain variant and average_precision columns.")

    keep_order = [
        "keyword_only",
        "rq_only",
        "proposal_only",
        "rq_proposal",
        "rq_keyword",
        "semantic_heavy",
        "balanced",
        "full_hybrid_original",
    ]
    variant_labels = {
        "keyword_only": "Keyword only",
        "rq_only": "RQ only",
        "proposal_only": "Proposal only",
        "rq_proposal": "RQ + Proposal",
        "rq_keyword": "RQ + Keyword",
        "semantic_heavy": "Semantic-heavy",
        "balanced": "Balanced",
        "full_hybrid_original": "Full hybrid",
    }
    # Highlight the configuration the paper ships with.
    emphasis = {"full_hybrid_original"}

    plot_df = df[df["variant"].isin(keep_order)].copy()
    plot_df["variant"] = pd.Categorical(plot_df["variant"], categories=keep_order, ordered=True)
    plot_df = plot_df.sort_values("variant")

    variants = plot_df["variant"].astype(str).tolist()
    colors = ["#D55E00" if v in emphasis else "#56B4E9" for v in variants]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(
        [variant_labels.get(v, v) for v in variants],
        plot_df["average_precision"].to_numpy(),
        color=colors, edgecolor="white", linewidth=0.6,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([variant_labels.get(v, v) for v in variants],
                       rotation=35, ha="right")
    ax.set_ylabel("Average precision")
    ax.set_title("Ablation comparison by average precision", loc="left")
    ax.set_ylim(0, max(plot_df["average_precision"]) * 1.15)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "ablation_average_precision")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    set_pub_style()

    plot_hybrid_score_distribution()
    plot_top100_composition()
    plot_recall_recovery_curve()
    plot_ablation_average_precision()


if __name__ == "__main__":
    main()
