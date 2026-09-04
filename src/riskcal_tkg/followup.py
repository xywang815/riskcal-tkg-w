from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCORE_NAMES = ("margin", "negscore", "minmax", "softmax")
ANSWER_COUNT_BINS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3_5", 3, 5),
    ("gt5", 6, None),
)


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

    result: dict[str, float | int] = {
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
        "vocabulary_size": int(mask.shape[1]),
        "mean_set_fraction": float(set_sizes.mean() / mask.shape[1]),
        "full_vocabulary_rate": float((set_sizes == mask.shape[1]).mean()),
    }
    for label, lower, upper in ANSWER_COUNT_BINS:
        selected = grouping.answer_counts >= lower
        if upper is not None:
            selected &= grouping.answer_counts <= upper
        count = int(selected.sum())
        result[f"answer_count_{label}_query_count"] = count
        result[f"answer_count_{label}_full_set_coverage"] = conditional_mean(
            full_set,
            selected,
        )
        result[f"answer_count_{label}_partial_answer_recall"] = conditional_mean(
            partial_recall,
            selected,
        )
        result[f"answer_count_{label}_mean_set_size"] = conditional_mean(
            set_sizes,
            selected,
        )
    return result


def summarize_budgeted_query_masks(
    query_mask: np.ndarray,
    query_scores: np.ndarray,
    facts: np.ndarray,
    grouping: QueryGrouping,
    budgets: tuple[int, ...],
) -> dict[str, float | int]:
    mask = np.asarray(query_mask, dtype=bool)
    scores = np.asarray(query_scores, dtype=float)
    if mask.shape != scores.shape or len(mask) != grouping.query_count:
        raise ValueError("query masks and scores must have one row per unique query")
    if not np.isfinite(scores).all():
        raise ValueError("query scores must be finite")
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must contain positive integers")
    result: dict[str, float | int] = {}
    base_sizes = mask.sum(axis=1)
    vocabulary_size = mask.shape[1]
    row_indices = np.arange(len(mask))[:, None]
    for budget in sorted(set(int(value) for value in budgets)):
        if budget >= vocabulary_size:
            budgeted = mask.copy()
        else:
            top_indices = np.argpartition(scores, -budget, axis=1)[:, -budget:]
            top_mask = np.zeros_like(mask)
            top_mask[row_indices, top_indices] = True
            budgeted = mask & top_mask
        summary = summarize_query_mask(budgeted, facts, grouping)
        prefix = f"budget_{budget}"
        result[f"{prefix}_label_coverage"] = summary["label_coverage"]
        result[f"{prefix}_full_set_coverage"] = summary["full_set_coverage"]
        result[f"{prefix}_partial_answer_recall"] = summary[
            "partial_answer_recall"
        ]
        result[f"{prefix}_mean_set_size"] = summary["mean_set_size"]
        result[f"{prefix}_fraction_truncated"] = float((base_sizes > budget).mean())
    return result
