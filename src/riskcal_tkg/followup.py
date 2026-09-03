from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCORE_NAMES = ("margin", "negscore", "minmax", "softmax")


def candidate_nonconformity(
    scores: np.ndarray,
    method: str,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite two-dimensional array")
    if method not in SCORE_NAMES:
        raise ValueError(f"unknown nonconformity method: {method}")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if method == "margin":
        return values.max(axis=1, keepdims=True) - values
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


@dataclass(frozen=True)
class QueryGrouping:
    first_indices: np.ndarray
    inverse: np.ndarray
    answer_counts: np.ndarray

    @property
    def query_count(self) -> int:
        return len(self.first_indices)


def build_query_grouping(facts: np.ndarray) -> QueryGrouping:
    values = np.asarray(facts, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError("facts must have shape (n, 4) and be nonempty")
    keys = values[:, [0, 1, 3]]
    _, first_indices, inverse, counts = np.unique(
        keys,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    return QueryGrouping(
        first_indices=np.asarray(first_indices, dtype=np.int64),
        inverse=np.asarray(inverse, dtype=np.int64),
        answer_counts=np.asarray(counts, dtype=np.int64),
    )


def query_max_true_nonconformity(
    candidate_values: np.ndarray,
    facts: np.ndarray,
    grouping: QueryGrouping,
) -> np.ndarray:
    values = np.asarray(candidate_values)
    labels = np.asarray(facts, dtype=np.int64)[:, 2]
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("candidate values and facts have incompatible shapes")
    true_values = values[np.arange(len(values)), labels]
    maxima = np.full(grouping.query_count, -np.inf, dtype=true_values.dtype)
    np.maximum.at(maxima, grouping.inverse, true_values)
    if not np.isfinite(maxima).all():
        raise ValueError("every query group must contain a finite true-label score")
    return maxima


def summarize_query_mask(
    query_mask: np.ndarray,
    facts: np.ndarray,
    grouping: QueryGrouping,
) -> dict[str, float | int]:
    mask = np.asarray(query_mask, dtype=bool)
    values = np.asarray(facts, dtype=np.int64)
    if mask.ndim != 2 or len(mask) != grouping.query_count:
        raise ValueError("query mask must have one row per unique query")
    labels = values[:, 2]
    if len(labels) and (labels.min() < 0 or labels.max() >= mask.shape[1]):
        raise ValueError("true entity ID out of range")

    label_covered = mask[grouping.inverse, labels]
    covered_counts = np.bincount(
        grouping.inverse,
        weights=label_covered.astype(float),
        minlength=grouping.query_count,
    )
    partial_recall = covered_counts / grouping.answer_counts
    full_set = covered_counts == grouping.answer_counts
    set_sizes = mask.sum(axis=1)
    single = grouping.answer_counts == 1
    multi = ~single

    def conditional_mean(values_: np.ndarray, selected: np.ndarray) -> float:
        return float(np.mean(values_[selected])) if np.any(selected) else float("nan")

    return {
        "label_count": int(len(labels)),
        "query_count": int(grouping.query_count),
        "single_query_count": int(single.sum()),
        "multi_query_count": int(multi.sum()),
        "label_coverage": float(label_covered.mean()),
        "full_set_coverage": float(full_set.mean()),
        "partial_answer_recall": float(partial_recall.mean()),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "p90_set_size": float(np.quantile(set_sizes, 0.9)),
        "single_full_set_coverage": conditional_mean(full_set, single),
        "multi_full_set_coverage": conditional_mean(full_set, multi),
        "single_mean_set_size": conditional_mean(set_sizes, single),
        "multi_mean_set_size": conditional_mean(set_sizes, multi),
    }

