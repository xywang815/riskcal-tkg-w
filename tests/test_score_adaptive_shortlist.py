import numpy as np
import pandas as pd
import pytest

from scripts.export_score_adaptive_shortlist import (
    MassCandidate,
    build_effect_frames,
    build_mass_candidates,
    bootstrap_effects,
    mass_nonconformity,
    mass_prediction_mask,
    select_mass_candidate,
)


def test_aps_nonconformity_matches_cumulative_softmax_mass() -> None:
    scores = np.asarray([[2.0, 1.0, 0.0]])
    candidate = MassCandidate("aps", 0, 0.0)
    value = mass_nonconformity(scores, np.asarray([1]), candidate)[0]

    exp = np.exp(np.asarray([0.0, -1.0, -2.0]))
    probabilities = exp / exp.sum()
    assert value == pytest.approx(float(probabilities[:2].sum()))


def test_raps_penalty_can_reduce_prediction_set_size() -> None:
    scores = np.asarray([[3.0, 2.0, 1.0, 0.0]])
    aps = MassCandidate("aps", 0, 0.0)
    raps = MassCandidate("raps_k1_lam0p25", 1, 0.25)

    aps_mask = mass_prediction_mask(scores, candidate=aps, threshold=0.90)
    raps_mask = mass_prediction_mask(scores, candidate=raps, threshold=0.90)

    assert aps_mask.sum() > raps_mask.sum()
    assert raps_mask[0, 0]


def test_candidate_builder_includes_aps_and_raps_grid() -> None:
    candidates = build_mass_candidates(k_values=(10, 20), penalties=(0.0, 0.001))

    assert candidates[0] == MassCandidate("aps", 0, 0.0)
    assert {candidate.name for candidate in candidates} == {
        "aps",
        "raps_k10_lam0p001",
        "raps_k20_lam0p001",
    }


def test_selection_prefers_smallest_feasible_validation_size() -> None:
    rows = pd.DataFrame(
        [
            {
                "candidate": "aps",
                "observed_label_coverage": 0.91,
                "full_set_coverage": 0.90,
                "mean_size": 500.0,
                "p90_size": 900.0,
            },
            {
                "candidate": "raps_small",
                "observed_label_coverage": 0.89,
                "full_set_coverage": 0.87,
                "mean_size": 100.0,
                "p90_size": 150.0,
            },
            {
                "candidate": "raps_feasible",
                "observed_label_coverage": 0.885,
                "full_set_coverage": 0.86,
                "mean_size": 80.0,
                "p90_size": 120.0,
            },
        ]
    )

    chosen = select_mass_candidate(
        rows,
        target_coverage=0.9,
        coverage_tolerance=0.02,
    )

    assert chosen["candidate"] == "raps_feasible"
    assert chosen["selection_feasible"]


def test_bootstrap_effects_compare_adaptive_against_two_baselines() -> None:
    records = []
    for seed in (17, 29):
        for timestamp in range(1, 5):
            for method, size, label_cov in (
                ("margin_rolling", 500.0, 0.91),
                ("rank_rolling", 400.0, 0.90),
                ("adaptive_mass_rolling", 250.0, 0.895),
            ):
                records.append(
                    {
                        "seed": seed,
                        "timestamp": timestamp,
                        "deletion_rate": 0.3,
                        "method": method,
                        "unique_query_count": 10,
                        "label_row_count": 20,
                        "observed_label_coverage": label_cov,
                        "full_set_coverage": label_cov - 0.01,
                        "mean_size": size,
                        "p90_size": size + 10,
                    }
                )
    rows = pd.DataFrame(records)

    effects = build_effect_frames(
        rows,
        deletion_rate=0.3,
        baseline_method="rank_rolling",
        candidate_method="adaptive_mass_rolling",
    )
    bootstrapped = bootstrap_effects(
        rows,
        deletion_rate=0.3,
        comparisons=(
            ("margin_rolling", "adaptive_mass_rolling"),
            ("rank_rolling", "adaptive_mass_rolling"),
        ),
        block_length=1,
        iterations=30,
        bootstrap_seed=5,
    )

    assert effects["mean_size_reduction"].mean() == pytest.approx(150.0)
    assert {
        "adaptive_mass_rolling_mean_size_reduction_vs_margin_rolling",
        "adaptive_mass_rolling_mean_size_reduction_vs_rank_rolling",
    } <= set(bootstrapped["statistic"])
