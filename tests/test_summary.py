from pathlib import Path
import json

import pandas as pd

from scripts.summarize_results import _weighted_means, evaluate_success_gate, summarize_run


def _supported_rows() -> pd.DataFrame:
    rows = []
    for seed in (17, 29, 43):
        for rate in (0.1, 0.2):
            rows.extend(
                [
                    {
                        "seed": seed,
                        "deletion_rate": rate,
                        "timestamp": 10,
                        "method": "static",
                        "coverage": 0.84,
                        "mean_size": 10.0,
                        "query_count": 20,
                        "mrr": 0.35,
                        "frequency_mrr": 0.20,
                        "risk_at_1": 0.2,
                        "answer_rate_at_1": 0.5,
                    },
                    {
                        "seed": seed,
                        "deletion_rate": rate,
                        "timestamp": 10,
                        "method": "rolling",
                        "coverage": 0.89,
                        "mean_size": 10.0,
                        "query_count": 20,
                        "mrr": 0.35,
                        "frequency_mrr": 0.20,
                        "risk_at_1": 0.1,
                        "answer_rate_at_1": 0.6,
                    },
                    {
                        "seed": seed,
                        "deletion_rate": rate,
                        "timestamp": 10,
                        "method": "weighted",
                        "coverage": 0.90,
                        "mean_size": 12.0,
                        "query_count": 20,
                        "mrr": 0.35,
                        "frequency_mrr": 0.20,
                        "risk_at_1": 0.08,
                        "answer_rate_at_1": 0.65,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _supported_queries(model_rank: int = 1, frequency_rank: int = 2) -> pd.DataFrame:
    rows = []
    for seed in (17, 29, 43):
        for rate in (0.1, 0.2):
            for timestamp in range(10, 30):
                rows.append(
                    {
                        "seed": seed,
                        "deletion_rate": rate,
                        "timestamp": timestamp,
                        "method": "static",
                        "rank": model_rank,
                        "frequency_rank": frequency_rank,
                    }
                )
    return pd.DataFrame(rows)


def test_success_gate_passes_preregistered_case() -> None:
    decision = evaluate_success_gate(
        _supported_rows(), target=0.90, query_rows=_supported_queries()
    )
    assert decision["coverage_conditions_met"] is True
    assert decision["direction_consistent"] is True
    assert decision["set_size_ratio_ok"] is True
    assert decision["scorer_better_than_frequency"] is True
    assert decision["supported"] is True


def test_success_gate_fails_when_weighted_sets_are_too_large() -> None:
    rows = _supported_rows()
    rows.loc[rows["method"] == "weighted", "mean_size"] = 20.0
    decision = evaluate_success_gate(
        rows, target=0.90, query_rows=_supported_queries()
    )
    assert decision["set_size_ratio_ok"] is False
    assert decision["supported"] is False


def test_success_gate_weights_window_coverage_by_query_count() -> None:
    rows = _supported_rows()
    rows["coverage"] = rows["coverage"].where(rows["method"] != "weighted", 0.90)
    later = rows.copy()
    rows["query_count"] = 99
    later["timestamp"] = 11
    later["coverage"] = later["coverage"].where(later["method"] != "weighted", 0.0)
    later["query_count"] = 1
    decision = evaluate_success_gate(
        pd.concat([rows, later], ignore_index=True),
        0.90,
        query_rows=_supported_queries(),
    )
    assert decision["coverage_conditions_met"] is True


def test_gap_must_hold_at_strongest_deletion_rate() -> None:
    rows = _supported_rows()
    strongest = rows["deletion_rate"] == rows["deletion_rate"].max()
    rows.loc[strongest & (rows["method"] == "static"), "coverage"] = 0.89
    decision = evaluate_success_gate(
        rows, target=0.90, query_rows=_supported_queries()
    )
    assert decision["gap_condition_met"] is False
    assert decision["supported"] is False


def test_frequency_baseline_is_a_required_gate() -> None:
    rows = _supported_rows()
    rows["frequency_mrr"] = 0.40
    decision = evaluate_success_gate(
        rows,
        target=0.90,
        query_rows=_supported_queries(model_rank=2, frequency_rank=1),
    )
    assert decision["scorer_better_than_frequency"] is False
    assert decision["supported"] is False


def test_frequency_test_requires_three_independent_seeds() -> None:
    queries = _supported_queries()
    queries = queries[queries["seed"] != 43]
    decision = evaluate_success_gate(
        _supported_rows(), target=0.90, query_rows=queries
    )
    assert decision["scorer_better_than_frequency"] is False


def test_plot_aggregation_weights_windows_by_query_count() -> None:
    rows = pd.DataFrame(
        [
            {"method": "weighted", "deletion_rate": 0.2, "query_count": 99, "mean_size": 10.0},
            {"method": "weighted", "deletion_rate": 0.2, "query_count": 1, "mean_size": 100.0},
        ]
    )
    grouped = _weighted_means(rows, ["method", "deletion_rate"], ["mean_size"])
    assert grouped.iloc[0]["mean_size"] == 10.9


def test_summarize_run_replaces_pending_gate_and_writes_figures(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    figures = tmp_path / "figures"
    metrics.mkdir()
    figures.mkdir()
    _supported_rows().to_csv(metrics / "per_window.csv", index=False)
    _supported_queries().to_parquet(metrics / "per_query.parquet", index=False)
    (tmp_path / "SUCCESS_GATE.json").write_text(
        '{"status": "pending_summary"}\n', encoding="utf-8"
    )
    summarize_run(tmp_path, target=0.90)
    gate = json.loads((tmp_path / "SUCCESS_GATE.json").read_text(encoding="utf-8"))
    assert gate["status"] == "evaluated"
    for stem in ("coverage_by_time", "set_size_by_corruption", "risk_coverage"):
        assert (figures / f"{stem}.png").stat().st_size > 0
        assert (figures / f"{stem}.pdf").stat().st_size > 0
