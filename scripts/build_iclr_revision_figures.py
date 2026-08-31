from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data" / "final_confirmatory"
OUT = ROOT / "paper" / "figures" / "iclr_revision"

COLORS = {
    "static": "#B05A45",
    "rolling": "#2F6B9A",
    "weighted": "#3A8B6E",
    "adaptive": "#8B5E9B",
    "margin_rolling": "#2F6B9A",
    "rank_rolling": "#D1872C",
    "aps_rolling": "#3A8B6E",
    "adaptive_mass_rolling": "#8B5E9B",
}

LABELS = {
    "static": "Static",
    "rolling": "Rolling-1000",
    "weighted": "Weighted-1000",
    "adaptive": "Selected half-life",
    "margin_rolling": "Margin rolling",
    "rank_rolling": "Rank rolling",
    "aps_rolling": "APS rolling",
    "adaptive_mass_rolling": "Validation-selected RAPS",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", va="top")
    ax.set_title(title, loc="left", pad=8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)


def build_overview() -> None:
    ranking = pd.read_csv(DATA / "ranking_table.csv")
    window = pd.read_csv(DATA / "window_ablation_summary.csv")
    relation = pd.read_csv(DATA / "relation_worst_group_paper_table.csv")

    primary = window[
        window["method"].isin(["static", "rolling_count_1000"])
    ].copy()
    primary["display_method"] = primary["method"].map(
        {"static": "static", "rolling_count_1000": "rolling"}
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2), constrained_layout=True)

    ax = axes[0, 0]
    x = ranking["deletion_rate"] * 100
    ax.plot(x, ranking["mrr"], marker="o", color="#2F6B9A", label="Temporal DistMult")
    ax.plot(x, ranking["frequency_mrr"], marker="s", color="#777777", label="Frequency")
    ax.set_xlabel("Deleted training facts (%)")
    ax.set_ylabel("Filtered MRR")
    ax.set_xticks([0, 10, 20, 30])
    ax.legend(frameon=False)
    _panel(ax, "(a)", "Ranking quality")

    ax = axes[0, 1]
    for method in ["static", "rolling"]:
        rows = primary[primary["display_method"] == method].sort_values("deletion_rate")
        ax.errorbar(
            rows["deletion_rate"] * 100,
            rows["coverage_mean"],
            yerr=rows["coverage_sd"],
            marker="o",
            capsize=2,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1, label="Target")
    ax.set_xlabel("Deleted training facts (%)")
    ax.set_ylabel("Observed-label coverage")
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0.79, 0.92)
    ax.legend(frameon=False, ncol=2)
    _panel(ax, "(b)", "Prequential reliability")

    ax = axes[1, 0]
    for method in ["static", "rolling"]:
        rows = primary[primary["display_method"] == method].sort_values("deletion_rate")
        ax.errorbar(
            rows["deletion_rate"] * 100,
            rows["mean_size_mean"],
            yerr=rows["mean_size_sd"],
            marker="o",
            capsize=2,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set_xlabel("Deleted training facts (%)")
    ax.set_ylabel("Mean prediction-set size")
    ax.set_xticks([0, 10, 20, 30])
    ax.legend(frameon=False)
    _panel(ax, "(c)", "Reliability cost")

    ax = axes[1, 1]
    for method in ["static", "rolling", "weighted", "adaptive"]:
        rows = relation[relation["method"] == method].sort_values("deletion_rate")
        ax.plot(
            rows["deletion_rate"] * 100,
            rows["relation_side_coverage_min"],
            marker="o",
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Deleted training facts (%)")
    ax.set_ylabel("Worst eligible group coverage")
    ax.set_xticks([0, 30])
    ax.set_ylim(0.58, 0.93)
    ax.legend(frameon=False, ncol=2)
    _panel(ax, "(d)", "Relation-side diagnostic")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "empirical_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "empirical_overview.png", bbox_inches="tight")
    plt.close(fig)


def build_utility_and_robustness() -> None:
    shortlist = pd.read_csv(DATA / "score_adaptive_shortlist_summary.csv")
    shortlist = shortlist[shortlist["deletion_rate"] == 0.3].copy()
    window = pd.read_csv(DATA / "window_ablation_summary.csv")
    window = window[window["deletion_rate"] == 0.3].copy()
    delay = pd.read_csv(DATA / "delay_feedback_summary.csv")
    delay = delay[(delay["deletion_rate"] == 0.3) & (delay["method"] == "rolling")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75), constrained_layout=True)

    ax = axes[0]
    for _, row in shortlist.iterrows():
        method = row["method"]
        ax.scatter(
            row["mean_size_mean"],
            row["observed_label_coverage_mean"],
            s=34,
            color=COLORS[method],
            label=LABELS[method],
            zorder=3,
        )
    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean set size")
    ax.set_ylabel("Observed-label coverage")
    ax.set_ylim(0.8955, 0.9015)
    ax.legend(frameon=False, fontsize=6.8)
    _panel(ax, "(a)", "Shortlist operating points")

    ax = axes[1]
    selected = window[
        window["method"].isin(
            [
                "static",
                "expanding",
                "rolling_count_250",
                "rolling_count_500",
                "rolling_count_1000",
                "rolling_count_2000",
                "time_window_3",
                "time_window_7",
                "time_window_14",
                "time_window_30",
            ]
        )
    ].copy()
    selected["family"] = np.where(
        selected["method"].str.startswith("rolling_count"),
        "Count window",
        np.where(selected["method"].str.startswith("time_window"), "Time window", "Reference"),
    )
    family_color = {"Count window": "#2F6B9A", "Time window": "#D1872C", "Reference": "#777777"}
    for family, rows in selected.groupby("family"):
        ax.scatter(
            rows["mean_size_mean"],
            rows["coverage_mean"],
            s=28,
            color=family_color[family],
            label=family,
            zorder=3,
        )
    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean set size")
    ax.set_ylabel("Observed-label coverage")
    ax.set_ylim(0.81, 0.925)
    ax.legend(frameon=False, fontsize=6.8)
    _panel(ax, "(b)", "Window ablation at 30% deletion")

    ax = axes[2]
    ax.errorbar(
        delay["extra_delay_blocks"],
        delay["coverage_mean"],
        yerr=delay["coverage_sd"],
        color="#2F6B9A",
        marker="o",
        capsize=2,
    )
    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Additional withheld batches")
    ax.set_ylabel("Observed-label coverage")
    ax.set_xticks([0, 1, 3, 7])
    ax.set_ylim(0.884, 0.903)
    _panel(ax, "(c)", "Delayed feedback")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "utility_robustness.pdf", bbox_inches="tight")
    fig.savefig(OUT / "utility_robustness.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    build_overview()
    build_utility_and_robustness()


if __name__ == "__main__":
    main()
