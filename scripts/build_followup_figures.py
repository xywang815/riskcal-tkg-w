#!/usr/bin/env python3
"""Build checksum-bound figures for the follow-up calibration analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CASE_ORDER = (
    "icews14_distmult",
    "icews14_continuous_complex",
    "icews05_15_distmult",
    "icews05_15_continuous_complex",
)
CASE_LABELS = {
    "icews14_distmult": "ICEWS14 | DistMult",
    "icews14_continuous_complex": "ICEWS14 | continuous complex",
    "icews05_15_distmult": "ICEWS05-15 | DistMult",
    "icews05_15_continuous_complex": "ICEWS05-15 | continuous complex",
}
SCORE_ORDER = ("margin", "negscore", "minmax", "softmax")
SCORE_LABELS = {
    "margin": "Margin",
    "negscore": "NegScore",
    "minmax": "Minmax",
    "softmax": "Softmax (T=1)",
}
SCORE_COLORS = {
    "margin": "#0072B2",
    "negscore": "#E69F00",
    "minmax": "#009E73",
    "softmax": "#CC79A7",
}
HISTORY_ORDER = ("static", "expanding", "rolling")
HISTORY_LABELS = {
    "static": "Static",
    "expanding": "Expanding",
    "rolling": "Rolling-1000",
}
HISTORY_MARKERS = {"static": "s", "expanding": "^", "rolling": "o"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksums(directory: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            records[name.strip()] = digest
    return records


def _verify(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    expected = _checksums(directory)
    verified: dict[str, str] = {}
    for name in names:
        path = directory / name
        if not path.is_file() or name not in expected:
            raise ValueError(f"missing checksum-bound input: {path}")
        observed = _sha256(path)
        if observed != expected[name]:
            raise ValueError(f"checksum mismatch: {path}")
        verified[name] = observed
    return verified


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "RiskCal-TKG", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(png, bbox_inches="tight", metadata={"Software": "RiskCal-TKG"})
    plt.close(fig)
    return {pdf.name: _sha256(pdf), png.name: _sha256(png)}


def build_factorial(by_seed: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    required = {
        "case",
        "objective",
        "score",
        "history",
        "label_coverage",
        "mean_set_size",
    }
    missing = sorted(required - set(by_seed.columns))
    if missing:
        raise ValueError(f"followup_by_seed.csv is missing columns: {missing}")

    rows = by_seed[by_seed["objective"].eq("label")].copy()
    means = (
        rows.groupby(["case", "score", "history"], as_index=False)[
            ["label_coverage", "mean_set_size"]
        ]
        .mean()
    )
    expected = len(CASE_ORDER) * len(SCORE_ORDER) * len(HISTORY_ORDER)
    if len(means) != expected or means.isna().any().any():
        raise ValueError("factorial figure requires a complete finite 4 x 4 x 3 grid")

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.25), sharey=True)
    for index, (ax, case) in enumerate(zip(axes.flat, CASE_ORDER, strict=True)):
        case_rows = means[means["case"].eq(case)]
        for score in SCORE_ORDER:
            score_rows = (
                case_rows[case_rows["score"].eq(score)]
                .set_index("history")
                .loc[list(HISTORY_ORDER)]
            )
            x = score_rows["mean_set_size"].to_numpy(dtype=float) / 1000.0
            y = score_rows["label_coverage"].to_numpy(dtype=float)
            color = SCORE_COLORS[score]
            ax.plot(x, y, color=color, linewidth=1.25, alpha=0.9, zorder=2)
            for history, x_value, y_value in zip(HISTORY_ORDER, x, y, strict=True):
                ax.scatter(
                    x_value,
                    y_value,
                    marker=HISTORY_MARKERS[history],
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    s=29,
                    zorder=3,
                )
        ax.axhline(0.90, color="#333333", linestyle="--", linewidth=0.9)
        ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.75)
        ax.set_ylim(0.775, 1.0)
        ax.set_title(f"({chr(97 + index)}) {CASE_LABELS[case]}", loc="left", pad=6)
    for ax in axes[1, :]:
        ax.set_xlabel("Mean set size (thousands of entities)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed-label coverage")

    score_handles = [
        Line2D([0], [0], color=SCORE_COLORS[s], lw=1.5, label=SCORE_LABELS[s])
        for s in SCORE_ORDER
    ]
    history_handles = [
        Line2D(
            [0],
            [0],
            marker=HISTORY_MARKERS[h],
            color="#555555",
            linestyle="None",
            markerfacecolor="#555555",
            markersize=5,
            label=HISTORY_LABELS[h],
        )
        for h in HISTORY_ORDER
    ]
    fig.legend(
        handles=score_handles + history_handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.7,
    )
    fig.subplots_adjust(top=0.86, hspace=0.32, wspace=0.18)
    return _save(fig, output_dir, "followup_factorial")


def _forest_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    panel: str,
    scale: float = 1.0,
    show_labels: bool = False,
) -> None:
    selected = rows[rows["metric"].eq(metric)].set_index("case").loc[list(CASE_ORDER)]
    y = np.arange(len(CASE_ORDER))
    observed = selected["observed"].to_numpy(dtype=float) * scale
    low = selected["ci95_low"].to_numpy(dtype=float) * scale
    high = selected["ci95_high"].to_numpy(dtype=float) * scale
    ax.errorbar(
        observed,
        y,
        xerr=np.vstack((observed - low, high - observed)),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        markersize=4.0,
        elinewidth=1.25,
        capsize=2.3,
    )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels([CASE_LABELS[c] for c in CASE_ORDER])
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    ax.set_title(f"({panel}) {title}", loc="left", pad=6)
    ax.set_xlabel(xlabel)


def build_query_objective(bootstrap: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    required = {
        "case",
        "family",
        "metric",
        "block_length",
        "observed",
        "ci95_low",
        "ci95_high",
    }
    missing = sorted(required - set(bootstrap.columns))
    if missing:
        raise ValueError(f"followup_bootstrap_summary.csv is missing columns: {missing}")
    rows = bootstrap[
        bootstrap["family"].eq("query_objective") & bootstrap["block_length"].eq(7)
    ].copy()
    if len(rows) != len(CASE_ORDER) * 3 or rows.isna().any().any():
        raise ValueError("query-objective figure requires 12 finite primary rows")

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65))
    _forest_panel(
        axes[0],
        rows,
        "full_set_coverage",
        "All answers covered",
        "Query-max minus label-level",
        "a",
        show_labels=True,
    )
    _forest_panel(
        axes[1],
        rows,
        "mean_set_size",
        "Mean set size",
        "Entity-count difference",
        "b",
    )
    _forest_panel(
        axes[2],
        rows,
        "multi_minus_single_full_set_coverage",
        "Multi-answer gap",
        "Gap change",
        "c",
    )
    fig.subplots_adjust(left=0.24, bottom=0.23, top=0.82, wspace=0.36)
    return _save(fig, output_dir, "followup_query_objective")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("paper/data/followup/analysis"))
    parser.add_argument("--bootstrap-root", type=Path, default=Path("paper/data/followup/bootstrap"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/followup"))
    args = parser.parse_args()

    analysis_names = ("followup_by_seed.csv",)
    bootstrap_names = ("followup_bootstrap_summary.csv",)
    inputs = {
        "analysis": _verify(args.analysis_root, analysis_names),
        "bootstrap": _verify(args.bootstrap_root, bootstrap_names),
    }
    _style()
    outputs: dict[str, str] = {}
    outputs.update(
        build_factorial(pd.read_csv(args.analysis_root / analysis_names[0]), args.output_dir)
    )
    outputs.update(
        build_query_objective(
            pd.read_csv(args.bootstrap_root / bootstrap_names[0]), args.output_dir
        )
    )

    manifest = {
        "inputs": inputs,
        "outputs": outputs,
        "primary_block_length": 7,
        "target_coverage": 0.90,
    }
    manifest_path = args.output_dir / "followup_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = args.output_dir / "SHA256SUMS.txt"
    checksum_records = dict(outputs)
    checksum_records[manifest_path.name] = _sha256(manifest_path)
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksum_records.items())),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
