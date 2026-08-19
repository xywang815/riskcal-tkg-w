import pandas as pd
import pytest

from scripts.export_score_adaptive_sensitivity import (
    aggregate_sensitivity_summary,
    bootstrap_sensitivity_effects,
    build_selection_summary,
    build_sensitivity_effects,
    build_success_table,
    summarize_sensitivity_rows,
)


def _toy_rows() -> pd.DataFrame:
    records = []
    for target in (0.88, 0.90):
        for tolerance in (0.0, 0.02):
            for seed in (17, 29):
                for timestamp in range(1, 5):
                    for method, mean_size, label_cov, full_cov in (
                        ("margin_rolling", 500.0, target + 0.010, target + 0.012),
                        ("rank_rolling", 400.0, target + 0.004, target - 0.002),
                        (
                            "adaptive_mass_rolling",
                            300.0 - 100.0 * tolerance,
                            target + 0.002,
                            target - 0.004,
                        ),
                    ):
                        records.append(
                            {
                                "target_coverage": target,
                                "selection_tolerance": tolerance,
                                "seed": seed,
                                "deletion_rate": 0.3,
                                "method": method,
                                "timestamp": timestamp,
                                "threshold": 1.0,
                                "threshold_units": "toy",
                                "pool_score_count": 100,
                                "pool_span_blocks": 4,
                                "label_row_count": 20,
                                "unique_query_count": 10,
                                "observed_label_coverage": label_cov,
                                "full_set_coverage": full_cov,
                                "partial_answer_recall": full_cov,
                                "mean_size": mean_size,
                                "median_size": mean_size - 10.0,
                                "p90_size": mean_size + 20.0,
                                "singleton_rate": 0.0,
                                "full_vocabulary_set_rate": 0.0,
                            }
                        )
    return pd.DataFrame(records)


def test_sensitivity_summary_keeps_target_and_tolerance_groups() -> None:
    rows = _toy_rows()

    by_seed = summarize_sensitivity_rows(rows)
    summary = aggregate_sensitivity_summary(by_seed)

    assert set(by_seed["target_coverage"]) == {0.88, 0.90}
    assert set(by_seed["selection_tolerance"]) == {0.0, 0.02}
    assert len(summary) == 12
    adaptive = summary[
        (summary["target_coverage"] == 0.90)
        & (summary["selection_tolerance"] == 0.02)
        & (summary["method"] == "adaptive_mass_rolling")
    ].iloc[0]
    assert adaptive["mean_size_mean"] == pytest.approx(298.0)


def test_sensitivity_effects_are_computed_per_grid_cell() -> None:
    rows = _toy_rows()

    effects = build_sensitivity_effects(
        rows,
        target_coverage=0.90,
        selection_tolerance=0.02,
        deletion_rate=0.3,
        baseline_method="rank_rolling",
        candidate_method="adaptive_mass_rolling",
    )

    assert effects["mean_size_reduction"].mean() == pytest.approx(102.0)
    assert effects["observed_label_coverage_delta"].mean() == pytest.approx(-0.002)


def test_selection_summary_keeps_grid_dimensions() -> None:
    rows = pd.DataFrame(
        [
            {
                "target_coverage": 0.88,
                "selection_tolerance": 0.0,
                "candidate": "aps",
                "k_reg": 0,
                "penalty": 0.0,
                "selected": True,
                "observed_label_coverage": 0.89,
                "full_set_coverage": 0.88,
                "mean_size": 500.0,
                "p90_size": 900.0,
            },
            {
                "target_coverage": 0.88,
                "selection_tolerance": 0.02,
                "candidate": "raps_k50_lam0p0001",
                "k_reg": 50,
                "penalty": 0.0001,
                "selected": True,
                "observed_label_coverage": 0.885,
                "full_set_coverage": 0.875,
                "mean_size": 300.0,
                "p90_size": 500.0,
            },
        ]
    )

    summary = build_selection_summary(rows)

    assert list(summary["selection_tolerance"]) == [0.0, 0.02]
    assert list(summary["condition_fraction"]) == [1.0, 1.0]


def test_bootstrap_and_success_table_cover_all_grid_cells() -> None:
    rows = _toy_rows()
    by_seed = summarize_sensitivity_rows(rows)
    summary = aggregate_sensitivity_summary(by_seed)

    bootstrapped = bootstrap_sensitivity_effects(
        rows,
        target_coverages=(0.88, 0.90),
        selection_tolerances=(0.0, 0.02),
        deletion_rate=0.3,
        comparisons=(
            ("margin_rolling", "adaptive_mass_rolling"),
            ("rank_rolling", "adaptive_mass_rolling"),
        ),
        block_length=1,
        iterations=30,
        bootstrap_seed=11,
    )
    success = build_success_table(
        summary,
        bootstrapped,
        target_coverages=(0.88, 0.90),
        selection_tolerances=(0.0, 0.02),
        deletion_rate=0.3,
    )

    assert len(success) == 4
    assert success["supported_vs_margin"].all()
    assert success["upgrade_over_rank_mean_supported"].all()
    assert {
        "adaptive_mass_rolling_mean_size_reduction_vs_margin_rolling",
        "adaptive_mass_rolling_mean_size_reduction_vs_rank_rolling",
    } <= set(bootstrapped["statistic"])
