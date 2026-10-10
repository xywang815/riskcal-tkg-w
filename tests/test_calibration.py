import numpy as np
import pytest

from riskcal_tkg.calibration import (
    AdaptiveHalfLifeSelector,
    CalibrationBatch,
    CalibrationPool,
    DriftHistory,
    choose_adaptive_half_life,
    drift_feature_vector,
    finite_sample_quantile,
    kgcp_nonconformity,
    kgcp_prediction_set_mask,
    margin_nonconformity,
    prediction_set_mask,
    select_half_life,
    select_half_life_with_validation,
    weighted_quantile,
    weighted_threshold,
)


def test_margin_and_inclusive_prediction_set() -> None:
    scores = np.asarray([[4.0, 3.0, 1.0]])
    assert margin_nonconformity(scores, np.asarray([1])).tolist() == [1.0]
    assert prediction_set_mask(scores, threshold=1.0).tolist() == [
        [True, True, False]
    ]


def test_finite_sample_quantile_uses_conformal_correction() -> None:
    assert finite_sample_quantile(np.asarray([0.0, 1.0, 2.0, 3.0]), alpha=0.25) == 3.0


def test_kgcp_negscore_matches_published_definition() -> None:
    scores = np.asarray([[4.0, 3.0, 1.0]])
    labels = np.asarray([1])
    nonconformity = kgcp_nonconformity(scores, labels, "negscore")
    assert nonconformity.tolist() == [-3.0]
    assert kgcp_prediction_set_mask(
        scores,
        threshold=-3.0,
        method="negscore",
    ).tolist() == [[True, True, False]]


def test_kgcp_minmax_and_softmax_include_the_calibrated_label() -> None:
    scores = np.asarray([[4.0, 3.0, 1.0], [2.0, 2.0, 2.0]])
    labels = np.asarray([1, 2])
    for method in ("minmax", "softmax"):
        nonconformity = kgcp_nonconformity(scores, labels, method)
        for index, threshold in enumerate(nonconformity):
            mask = kgcp_prediction_set_mask(
                scores[index : index + 1],
                threshold=float(threshold),
                method=method,
            )
            assert mask[0, labels[index]]
    degenerate_threshold = float(
        kgcp_nonconformity(scores[1:], labels[1:], "minmax")[0]
    )
    assert kgcp_prediction_set_mask(
        scores[1:],
        degenerate_threshold,
        "minmax",
    ).all()


def test_weighted_quantile_matches_hand_calculation() -> None:
    values = np.asarray([1.0, 2.0, 5.0])
    weights = np.asarray([0.1, 0.2, 0.7])
    assert weighted_quantile(values, weights, probability=0.8) == 5.0


def test_pool_rejects_current_or_future_labels() -> None:
    pool = CalibrationPool(max_size=10)
    pool.add(timestamp=2, scores=np.asarray([0.5]))
    with pytest.raises(ValueError, match="strictly earlier"):
        pool.values_before(timestamp=2)


def test_weighted_threshold_prefers_recent_values() -> None:
    pool = CalibrationPool(max_size=10)
    pool.add(timestamp=1, scores=np.asarray([10.0]))
    pool.add(timestamp=9, scores=np.asarray([1.0]))
    threshold = weighted_threshold(
        pool,
        timestamp=10,
        alpha=0.5,
        half_life=1.0,
    )
    assert threshold == 1.0


def test_half_life_selection_uses_early_history_and_later_validation() -> None:
    high_margin = np.tile(np.asarray([[10.0, 5.0, 0.0, 0.0, 0.0, 0.0]]), (20, 1))
    low_margin = np.tile(np.asarray([[10.0, 9.0, 0.0, 0.0, 0.0, 0.0]]), (20, 1))
    validation = np.tile(np.asarray([[10.0, 9.5, 8.0, 7.0, 6.0, 5.0]]), (20, 1))
    labels = np.ones(20, dtype=np.int64)
    batches = [
        CalibrationBatch(0, high_margin, labels),
        CalibrationBatch(9, low_margin, labels),
        CalibrationBatch(10, low_margin, labels),
        CalibrationBatch(11, validation, labels),
        CalibrationBatch(12, validation, labels),
    ]
    selection = select_half_life(
        batches,
        candidates=(1.0, float("inf")),
        target_coverage=0.9,
        max_size=1000,
    )
    assert selection.selected_half_life == 1.0
    assert selection.evaluations[0]["mean_size"] < selection.evaluations[1]["mean_size"]


def test_half_life_selection_can_use_explicit_validation_role() -> None:
    high_margin = np.tile(np.asarray([[10.0, 5.0, 0.0, 0.0]]), (10, 1))
    low_margin = np.tile(np.asarray([[10.0, 9.0, 0.0, 0.0]]), (10, 1))
    labels = np.ones(10, dtype=np.int64)
    selection = select_half_life_with_validation(
        history_batches=[CalibrationBatch(1, high_margin, labels)],
        validation_batches=[CalibrationBatch(2, low_margin, labels)],
        candidates=(1.0, float("inf")),
        target_coverage=0.9,
        max_size=100,
    )
    assert selection.selected_half_life in {1.0, float("inf")}
    assert len(selection.evaluations) == 2


def test_drift_history_does_not_use_current_or_past_true_labels() -> None:
    scores = np.asarray([[4.0, 3.0, 1.0], [1.0, 2.0, 3.0]])
    subjects = np.asarray([0, 1])
    relations = np.asarray([0, 1])
    first = CalibrationBatch(1, scores, np.asarray([0, 2]), subjects, relations)
    relabeled = CalibrationBatch(1, scores, np.asarray([1, 1]), subjects, relations)
    first_history = DriftHistory(num_relations=3)
    second_history = DriftHistory(num_relations=3)
    first_history.add_batch(first)
    second_history.add_batch(relabeled)
    current_scores = np.asarray([[3.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    current_subjects = np.asarray([0, 2])
    current_relations = np.asarray([0, 2])
    assert np.array_equal(
        drift_feature_vector(
            current_scores,
            current_subjects,
            current_relations,
            first_history.reference(),
        ),
        drift_feature_vector(
            current_scores,
            current_subjects,
            current_relations,
            second_history.reference(),
        ),
    )


def test_adaptive_selector_prefers_smaller_set_when_coverage_is_feasible() -> None:
    selector = AdaptiveHalfLifeSelector(
        candidates=(1.0, float("inf")),
        target_coverage=0.9,
        coverage_tolerance=0.02,
        num_relations=2,
        feature_mean=np.zeros(4),
        feature_scale=np.ones(4),
        coverage_coefficients={
            1.0: np.asarray([0.91, 0.0, 0.0, 0.0, 0.0]),
            float("inf"): np.asarray([0.91, 0.0, 0.0, 0.0, 0.0]),
        },
        log_size_coefficients={
            1.0: np.asarray([np.log1p(2.0), 0.0, 0.0, 0.0, 0.0]),
            float("inf"): np.asarray([np.log1p(5.0), 0.0, 0.0, 0.0, 0.0]),
        },
        fallback_half_life=float("inf"),
        fit_samples=3,
    )
    decision = choose_adaptive_half_life(selector, np.zeros(4))
    assert decision.half_life == 1.0
    assert decision.coverage_feasible
