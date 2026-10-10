from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


DRIFT_FEATURE_NAMES = (
    "relation_tv",
    "score_gap_ks",
    "novelty_rate",
    "log_query_count",
)

KGCP_SCORE_METHODS = ("negscore", "minmax", "softmax")


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("scores must be finite")
    return array


def _validate_id_vector(
    values: np.ndarray | None,
    length: int,
    *,
    name: str,
    upper_bound: int | None = None,
) -> np.ndarray:
    if values is None:
        return np.zeros(length, dtype=np.int64)
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have length {length}")
    if len(array) and array.min() < 0:
        raise ValueError(f"{name} must be nonnegative")
    if upper_bound is not None and len(array) and array.max() >= upper_bound:
        raise ValueError(f"{name} contains an out-of-range ID")
    return array


def margin_nonconformity(scores: np.ndarray, true_ids: np.ndarray) -> np.ndarray:
    values = _validate_scores(scores)
    labels = np.asarray(true_ids, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("scores and true_ids have incompatible shapes")
    if len(labels) and (labels.min() < 0 or labels.max() >= values.shape[1]):
        raise ValueError("true entity ID out of range")
    return values.max(axis=1) - values[np.arange(len(values)), labels]


def _kgcp_candidate_nonconformity(
    scores: np.ndarray,
    method: str,
    temperature: float = 1.0,
) -> np.ndarray:
    values = _validate_scores(scores)
    if values.ndim != 2:
        raise ValueError("scores must have shape (n, classes)")
    if method not in KGCP_SCORE_METHODS:
        raise ValueError(f"unknown KGCP score method: {method}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if method == "negscore":
        return -values
    if method == "minmax":
        minima = values.min(axis=1, keepdims=True)
        ranges = values.max(axis=1, keepdims=True) - minima
        normalized = np.divide(
            values - minima,
            ranges,
            out=np.zeros_like(values),
            where=ranges > 0,
        )
        return -normalized
    logits = values / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return 1.0 - probabilities


def kgcp_nonconformity(
    scores: np.ndarray,
    true_ids: np.ndarray,
    method: str,
    temperature: float = 1.0,
) -> np.ndarray:
    candidates = _kgcp_candidate_nonconformity(scores, method, temperature)
    labels = np.asarray(true_ids, dtype=np.int64)
    if labels.shape != (len(candidates),):
        raise ValueError("scores and true_ids have incompatible shapes")
    if len(labels) and (labels.min() < 0 or labels.max() >= candidates.shape[1]):
        raise ValueError("true entity ID out of range")
    return candidates[np.arange(len(candidates)), labels]


def kgcp_prediction_set_mask(
    scores: np.ndarray,
    threshold: float,
    method: str,
    temperature: float = 1.0,
) -> np.ndarray:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    candidates = _kgcp_candidate_nonconformity(scores, method, temperature)
    return candidates <= threshold


def prediction_set_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    values = _validate_scores(scores)
    if values.ndim != 2:
        raise ValueError("scores must have shape (n, classes)")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return values >= (values.max(axis=1, keepdims=True) - threshold)


def score_margin_gaps(scores: np.ndarray) -> np.ndarray:
    values = _validate_scores(scores)
    if values.ndim != 2:
        raise ValueError("scores must have shape (n, classes)")
    if values.shape[1] < 2:
        return np.zeros(len(values), dtype=float)
    top_two = np.partition(values, kth=-2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


def relation_distribution(relations: np.ndarray, num_relations: int) -> np.ndarray:
    if num_relations <= 0:
        raise ValueError("num_relations must be positive")
    relation_ids = _validate_id_vector(
        relations,
        len(np.asarray(relations).reshape(-1)),
        name="relations",
        upper_bound=num_relations,
    )
    counts = np.bincount(relation_ids, minlength=num_relations).astype(float)
    total = counts.sum()
    if total == 0:
        return np.zeros(num_relations, dtype=float)
    return counts / total


def total_variation_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float).reshape(-1)
    right_array = np.asarray(right, dtype=float).reshape(-1)
    if left_array.shape != right_array.shape:
        raise ValueError("distributions must have the same shape")
    if np.any(left_array < 0) or np.any(right_array < 0):
        raise ValueError("distributions must be nonnegative")
    return float(0.5 * np.abs(left_array - right_array).sum())


def empirical_ks_distance(values: np.ndarray, reference: np.ndarray) -> float:
    sample = np.sort(_validate_scores(values).reshape(-1))
    baseline = np.sort(_validate_scores(reference).reshape(-1))
    if len(sample) == 0 or len(baseline) == 0:
        return 0.0
    grid = np.sort(np.concatenate((sample, baseline)))
    sample_cdf = np.searchsorted(sample, grid, side="right") / len(sample)
    baseline_cdf = np.searchsorted(baseline, grid, side="right") / len(baseline)
    return float(np.max(np.abs(sample_cdf - baseline_cdf)))


def drift_feature_vector(
    scores: np.ndarray,
    subjects: np.ndarray | None,
    relations: np.ndarray | None,
    reference: DriftReference,
) -> np.ndarray:
    values = _validate_scores(scores)
    if values.ndim != 2:
        raise ValueError("scores must have shape (n, classes)")
    subject_ids = _validate_id_vector(subjects, len(values), name="subjects")
    relation_ids = _validate_id_vector(
        relations,
        len(values),
        name="relations",
        upper_bound=reference.num_relations,
    )
    current_distribution = relation_distribution(relation_ids, reference.num_relations)
    seen = sum(
        (int(subject), int(relation)) in reference.subject_relation_pairs
        for subject, relation in zip(subject_ids, relation_ids, strict=True)
    )
    novelty_rate = 0.0 if len(values) == 0 else 1.0 - (seen / len(values))
    return np.asarray(
        [
            total_variation_distance(
                current_distribution,
                reference.relation_distribution,
            ),
            empirical_ks_distance(score_margin_gaps(values), reference.score_gaps),
            novelty_rate,
            math.log1p(len(values)),
        ],
        dtype=float,
    )


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(_validate_scores(values).reshape(-1))
    if len(ordered) == 0:
        raise ValueError("calibration values must not be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1])


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    value_array = _validate_scores(values).reshape(-1)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if len(value_array) == 0 or len(value_array) != len(weight_array):
        raise ValueError("values and weights must have equal nonzero length")
    if (
        np.any(weight_array < 0)
        or not np.isfinite(weight_array).all()
        or weight_array.sum() <= 0
    ):
        raise ValueError("weights must be finite, nonnegative, and have positive sum")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    order = np.argsort(value_array, kind="stable")
    cumulative = np.cumsum(weight_array[order]) / weight_array.sum()
    index = min(
        int(np.searchsorted(cumulative, probability, side="left")),
        len(order) - 1,
    )
    return float(value_array[order[index]])


@dataclass(frozen=True)
class CalibrationSlice:
    values: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class CalibrationBatch:
    timestamp: int
    scores: np.ndarray
    true_ids: np.ndarray
    subjects: np.ndarray | None = None
    relations: np.ndarray | None = None


@dataclass(frozen=True)
class HalfLifeSelection:
    selected_half_life: float
    evaluations: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class DriftReference:
    num_relations: int
    relation_distribution: np.ndarray
    score_gaps: np.ndarray
    subject_relation_pairs: frozenset[tuple[int, int]]


@dataclass
class DriftHistory:
    num_relations: int
    _relations: list[np.ndarray] = field(default_factory=list, init=False)
    _score_gaps: list[np.ndarray] = field(default_factory=list, init=False)
    _subject_relation_pairs: set[tuple[int, int]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.num_relations <= 0:
            raise ValueError("num_relations must be positive")

    def add(
        self,
        scores: np.ndarray,
        subjects: np.ndarray | None = None,
        relations: np.ndarray | None = None,
    ) -> None:
        values = _validate_scores(scores)
        if values.ndim != 2:
            raise ValueError("scores must have shape (n, classes)")
        subject_ids = _validate_id_vector(subjects, len(values), name="subjects")
        relation_ids = _validate_id_vector(
            relations,
            len(values),
            name="relations",
            upper_bound=self.num_relations,
        )
        self._relations.append(relation_ids.copy())
        self._score_gaps.append(score_margin_gaps(values))
        self._subject_relation_pairs.update(
            (int(subject), int(relation))
            for subject, relation in zip(subject_ids, relation_ids, strict=True)
        )

    def add_batch(self, batch: CalibrationBatch) -> None:
        self.add(batch.scores, batch.subjects, batch.relations)

    def reference(self) -> DriftReference:
        if not self._score_gaps:
            raise ValueError("drift history is empty")
        relations = np.concatenate(self._relations)
        return DriftReference(
            num_relations=self.num_relations,
            relation_distribution=relation_distribution(relations, self.num_relations),
            score_gaps=np.concatenate(self._score_gaps),
            subject_relation_pairs=frozenset(self._subject_relation_pairs),
        )


@dataclass(frozen=True)
class AdaptiveHalfLifeDecision:
    half_life: float
    predicted_coverage: float
    predicted_mean_size: float
    coverage_feasible: bool


@dataclass(frozen=True)
class AdaptiveHalfLifeSelector:
    candidates: tuple[float, ...]
    target_coverage: float
    coverage_tolerance: float
    num_relations: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coverage_coefficients: dict[float, np.ndarray]
    log_size_coefficients: dict[float, np.ndarray]
    fallback_half_life: float
    fit_samples: int


@dataclass
class CalibrationPool:
    max_size: int
    _timestamps: list[np.ndarray] = field(default_factory=list, init=False)
    _scores: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")

    def add(self, timestamp: int, scores: np.ndarray) -> None:
        values = _validate_scores(scores).reshape(-1)
        if len(values) == 0:
            raise ValueError("scores must not be empty")
        self._timestamps.append(np.full(len(values), int(timestamp), dtype=np.int64))
        self._scores.append(values.copy())
        total = sum(len(batch) for batch in self._scores)
        while total > self.max_size and self._scores:
            overflow = total - self.max_size
            if overflow >= len(self._scores[0]):
                total -= len(self._scores.pop(0))
                self._timestamps.pop(0)
            else:
                self._scores[0] = self._scores[0][overflow:]
                self._timestamps[0] = self._timestamps[0][overflow:]
                total -= overflow

    def values_before(self, timestamp: int) -> CalibrationSlice:
        if not self._scores:
            raise ValueError("calibration pool is empty")
        timestamps = np.concatenate(self._timestamps)
        if np.any(timestamps >= timestamp):
            raise ValueError("calibration labels must be strictly earlier than query time")
        return CalibrationSlice(np.concatenate(self._scores), timestamps)


def static_threshold(values: np.ndarray, alpha: float) -> float:
    return finite_sample_quantile(values, alpha)


def rolling_threshold(
    pool: CalibrationPool,
    timestamp: int,
    alpha: float,
) -> float:
    return finite_sample_quantile(pool.values_before(timestamp).values, alpha)


def weighted_threshold(
    pool: CalibrationPool,
    timestamp: int,
    alpha: float,
    half_life: float,
) -> float:
    calibration = pool.values_before(timestamp)
    if math.isinf(half_life):
        weights = np.ones_like(calibration.values)
    else:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        ages = timestamp - calibration.timestamps
        weights = np.exp2(-ages / half_life)
    return weighted_quantile(calibration.values, weights, 1.0 - alpha)


def select_half_life(
    batches: list[CalibrationBatch],
    candidates: tuple[float, ...],
    target_coverage: float,
    max_size: int,
    history_fraction: float = 0.6,
) -> HalfLifeSelection:
    if len(batches) < 2:
        raise ValueError("half-life selection requires at least two timestamp batches")
    if not candidates or any(value <= 0 or math.isnan(value) for value in candidates):
        raise ValueError("half-life candidates must be positive")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    if not 0.0 < history_fraction < 1.0:
        raise ValueError("history_fraction must be between 0 and 1")
    ordered = sorted(batches, key=lambda batch: batch.timestamp)
    if len({batch.timestamp for batch in ordered}) != len(ordered):
        raise ValueError("calibration batches must have unique timestamps")
    split_at = max(1, int(math.floor(history_fraction * len(ordered))))
    split_at = min(split_at, len(ordered) - 1)
    history, validation = ordered[:split_at], ordered[split_at:]
    evaluations: list[dict[str, float]] = []
    alpha = 1.0 - target_coverage

    for candidate in candidates:
        pool = CalibrationPool(max_size=max_size)
        for batch in history:
            pool.add(
                batch.timestamp,
                margin_nonconformity(batch.scores, batch.true_ids),
            )
        covered = 0
        set_size_total = 0
        query_count = 0
        for batch in validation:
            threshold = weighted_threshold(pool, batch.timestamp, alpha, candidate)
            mask = prediction_set_mask(batch.scores, threshold)
            labels = np.asarray(batch.true_ids, dtype=np.int64)
            covered += int(mask[np.arange(len(mask)), labels].sum())
            set_size_total += int(mask.sum())
            query_count += len(labels)
            pool.add(
                batch.timestamp,
                margin_nonconformity(batch.scores, labels),
            )
        evaluations.append(
            {
                "half_life": float(candidate),
                "coverage": covered / query_count,
                "mean_size": set_size_total / query_count,
                "query_count": float(query_count),
            }
        )

    lower = target_coverage - 0.02
    upper = target_coverage + 0.02
    qualified = [
        result for result in evaluations if lower <= result["coverage"] <= upper
    ]
    if qualified:
        chosen = min(qualified, key=lambda result: result["mean_size"])
    else:
        chosen = min(
            evaluations,
            key=lambda result: (
                abs(result["coverage"] - target_coverage),
                result["mean_size"],
            ),
        )
    return HalfLifeSelection(
        selected_half_life=chosen["half_life"],
        evaluations=tuple(evaluations),
    )


def select_half_life_with_validation(
    history_batches: list[CalibrationBatch],
    validation_batches: list[CalibrationBatch],
    candidates: tuple[float, ...],
    target_coverage: float,
    max_size: int,
) -> HalfLifeSelection:
    if not history_batches or not validation_batches:
        raise ValueError("history and validation batches must be nonempty")
    if not candidates or any(value <= 0 or math.isnan(value) for value in candidates):
        raise ValueError("half-life candidates must be positive")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    history = sorted(history_batches, key=lambda batch: batch.timestamp)
    validation = sorted(validation_batches, key=lambda batch: batch.timestamp)
    timestamps = [batch.timestamp for batch in history + validation]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("calibration batches must have unique timestamps")
    if history[-1].timestamp >= validation[0].timestamp:
        raise ValueError("history batches must be earlier than validation batches")

    evaluations: list[dict[str, float]] = []
    alpha = 1.0 - target_coverage
    for candidate in candidates:
        pool = CalibrationPool(max_size=max_size)
        for batch in history:
            pool.add(
                batch.timestamp,
                margin_nonconformity(batch.scores, batch.true_ids),
            )
        covered = 0
        set_size_total = 0
        query_count = 0
        for batch in validation:
            threshold = weighted_threshold(pool, batch.timestamp, alpha, candidate)
            mask = prediction_set_mask(batch.scores, threshold)
            labels = np.asarray(batch.true_ids, dtype=np.int64)
            covered += int(mask[np.arange(len(mask)), labels].sum())
            set_size_total += int(mask.sum())
            query_count += len(labels)
            pool.add(batch.timestamp, margin_nonconformity(batch.scores, labels))
        evaluations.append(
            {
                "half_life": float(candidate),
                "coverage": covered / query_count,
                "mean_size": set_size_total / query_count,
                "query_count": float(query_count),
            }
        )

    lower = target_coverage - 0.02
    upper = target_coverage + 0.02
    qualified = [
        result for result in evaluations if lower <= result["coverage"] <= upper
    ]
    if qualified:
        chosen = min(qualified, key=lambda result: result["mean_size"])
    else:
        chosen = min(
            evaluations,
            key=lambda result: (
                abs(result["coverage"] - target_coverage),
                result["mean_size"],
            ),
        )
    return HalfLifeSelection(
        selected_half_life=chosen["half_life"],
        evaluations=tuple(evaluations),
    )


def _ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    design = np.column_stack((np.ones(len(features)), features))
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def _selector_design(selector: AdaptiveHalfLifeSelector, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(features, dtype=float).reshape(-1)
    if raw.shape != selector.feature_mean.shape:
        raise ValueError("feature vector has the wrong shape")
    normalized = (raw - selector.feature_mean) / selector.feature_scale
    return np.concatenate((np.asarray([1.0]), normalized))


def choose_adaptive_half_life(
    selector: AdaptiveHalfLifeSelector,
    features: np.ndarray,
) -> AdaptiveHalfLifeDecision:
    design = _selector_design(selector, features)
    predictions: list[tuple[float, float, float, bool]] = []
    for candidate in selector.candidates:
        coverage = float(design @ selector.coverage_coefficients[candidate])
        mean_size = math.expm1(float(design @ selector.log_size_coefficients[candidate]))
        coverage = min(1.0, max(0.0, coverage))
        mean_size = max(0.0, mean_size)
        feasible = coverage >= selector.target_coverage - selector.coverage_tolerance
        predictions.append((candidate, coverage, mean_size, feasible))

    feasible_predictions = [item for item in predictions if item[3]]
    if feasible_predictions:
        chosen = min(
            feasible_predictions,
            key=lambda item: (
                item[2],
                abs(item[1] - selector.target_coverage),
                selector.candidates.index(item[0]),
            ),
        )
    else:
        chosen = max(
            predictions,
            key=lambda item: (
                item[1],
                -item[2],
                -selector.candidates.index(item[0]),
            ),
        )
    return AdaptiveHalfLifeDecision(
        half_life=chosen[0],
        predicted_coverage=chosen[1],
        predicted_mean_size=chosen[2],
        coverage_feasible=chosen[3],
    )


def fit_adaptive_half_life_selector(
    batches: list[CalibrationBatch],
    candidates: tuple[float, ...],
    target_coverage: float,
    max_size: int,
    num_relations: int,
    ridge: float = 1.0,
    coverage_tolerance: float = 0.02,
) -> AdaptiveHalfLifeSelector:
    if len(batches) < 2:
        raise ValueError("adaptive half-life selection requires at least two batches")
    if not candidates or any(value <= 0 or math.isnan(value) for value in candidates):
        raise ValueError("half-life candidates must be positive")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    ordered = sorted(batches, key=lambda batch: batch.timestamp)
    if len({batch.timestamp for batch in ordered}) != len(ordered):
        raise ValueError("calibration batches must have unique timestamps")

    alpha = 1.0 - target_coverage
    pool = CalibrationPool(max_size=max_size)
    history = DriftHistory(num_relations=num_relations)
    first = ordered[0]
    pool.add(first.timestamp, margin_nonconformity(first.scores, first.true_ids))
    history.add_batch(first)

    feature_rows: list[np.ndarray] = []
    coverage_targets: dict[float, list[float]] = {candidate: [] for candidate in candidates}
    size_targets: dict[float, list[float]] = {candidate: [] for candidate in candidates}
    for batch in ordered[1:]:
        features = drift_feature_vector(
            batch.scores,
            batch.subjects,
            batch.relations,
            history.reference(),
        )
        labels = np.asarray(batch.true_ids, dtype=np.int64)
        feature_rows.append(features)
        for candidate in candidates:
            threshold = weighted_threshold(pool, batch.timestamp, alpha, candidate)
            mask = prediction_set_mask(batch.scores, threshold)
            coverage_targets[candidate].append(
                float(mask[np.arange(len(mask)), labels].mean())
            )
            size_targets[candidate].append(float(mask.sum(axis=1).mean()))
        pool.add(batch.timestamp, margin_nonconformity(batch.scores, labels))
        history.add_batch(batch)

    raw_features = np.vstack(feature_rows)
    feature_mean = raw_features.mean(axis=0)
    feature_scale = raw_features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized_features = (raw_features - feature_mean) / feature_scale
    coverage_coefficients: dict[float, np.ndarray] = {}
    log_size_coefficients: dict[float, np.ndarray] = {}
    for candidate in candidates:
        coverage_coefficients[candidate] = _ridge_fit(
            normalized_features,
            np.asarray(coverage_targets[candidate], dtype=float),
            ridge,
        )
        log_size_coefficients[candidate] = _ridge_fit(
            normalized_features,
            np.log1p(np.asarray(size_targets[candidate], dtype=float)),
            ridge,
        )

    fallback = select_half_life(
        ordered,
        candidates=candidates,
        target_coverage=target_coverage,
        max_size=max_size,
    ).selected_half_life
    return AdaptiveHalfLifeSelector(
        candidates=tuple(candidates),
        target_coverage=target_coverage,
        coverage_tolerance=coverage_tolerance,
        num_relations=num_relations,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coverage_coefficients=coverage_coefficients,
        log_size_coefficients=log_size_coefficients,
        fallback_half_life=fallback,
        fit_samples=len(raw_features),
    )


def evaluate_adaptive_half_life_selector(
    selector: AdaptiveHalfLifeSelector,
    batches: list[CalibrationBatch],
    max_size: int,
    warmup_batches: list[CalibrationBatch] | None = None,
) -> tuple[dict[str, float], ...]:
    ordered = sorted(batches, key=lambda batch: batch.timestamp)
    warmup = sorted(warmup_batches or [], key=lambda batch: batch.timestamp)
    if not ordered:
        return ()
    pool = CalibrationPool(max_size=max_size)
    history = DriftHistory(num_relations=selector.num_relations)
    for batch in warmup:
        pool.add(batch.timestamp, margin_nonconformity(batch.scores, batch.true_ids))
        history.add_batch(batch)
    if not warmup:
        first = ordered.pop(0)
        pool.add(first.timestamp, margin_nonconformity(first.scores, first.true_ids))
        history.add_batch(first)

    rows: list[dict[str, float]] = []
    alpha = 1.0 - selector.target_coverage
    for batch in ordered:
        features = drift_feature_vector(
            batch.scores,
            batch.subjects,
            batch.relations,
            history.reference(),
        )
        decision = choose_adaptive_half_life(selector, features)
        threshold = weighted_threshold(pool, batch.timestamp, alpha, decision.half_life)
        mask = prediction_set_mask(batch.scores, threshold)
        labels = np.asarray(batch.true_ids, dtype=np.int64)
        rows.append(
            {
                "timestamp": float(batch.timestamp),
                "half_life": float(decision.half_life),
                "coverage": float(mask[np.arange(len(mask)), labels].mean()),
                "mean_size": float(mask.sum(axis=1).mean()),
                "predicted_coverage": decision.predicted_coverage,
                "predicted_mean_size": decision.predicted_mean_size,
            }
        )
        pool.add(batch.timestamp, margin_nonconformity(batch.scores, labels))
        history.add_batch(batch)
    return tuple(rows)
