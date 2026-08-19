from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoverageResult:
    coverage: float
    mean_size: float
    median_size: float
    p90_size: float


@dataclass(frozen=True)
class SelectiveResult:
    answer_rate: float
    abstention_rate: float
    risk: float


@dataclass(frozen=True)
class RankingResult:
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float


def filtered_rank(
    scores: np.ndarray,
    true_id: int,
    other_true_ids: set[int] | None = None,
) -> int:
    values = np.asarray(scores, dtype=float).reshape(-1).copy()
    if not np.isfinite(values).all() or not 0 <= true_id < len(values):
        raise ValueError("scores must be finite and true_id must be valid")
    for entity_id in other_true_ids or set():
        if entity_id != true_id and 0 <= entity_id < len(values):
            values[entity_id] = -np.inf
    order = np.lexsort((np.arange(len(values)), -values))
    return int(np.flatnonzero(order == true_id)[0] + 1)


def coverage_and_size(mask: np.ndarray, true_ids: np.ndarray) -> CoverageResult:
    values = np.asarray(mask, dtype=bool)
    labels = np.asarray(true_ids, dtype=np.int64)
    if values.ndim != 2 or len(values) == 0 or labels.shape != (len(values),):
        raise ValueError("mask and true_ids have incompatible or empty shapes")
    if labels.min() < 0 or labels.max() >= values.shape[1]:
        raise ValueError("true entity ID out of range")
    sizes = values.sum(axis=1)
    covered = values[np.arange(len(values)), labels]
    return CoverageResult(
        coverage=float(covered.mean()),
        mean_size=float(sizes.mean()),
        median_size=float(np.median(sizes)),
        p90_size=float(np.quantile(sizes, 0.9)),
    )


def selective_metrics(
    sizes: np.ndarray,
    top1_correct: np.ndarray,
    max_set_size: int,
) -> SelectiveResult:
    set_sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    correct = np.asarray(top1_correct, dtype=bool).reshape(-1)
    if len(set_sizes) == 0 or len(set_sizes) != len(correct):
        raise ValueError("sizes and correctness must have equal nonzero length")
    if max_set_size < 0:
        raise ValueError("max_set_size must be nonnegative")
    answered = set_sizes <= max_set_size
    answer_rate = float(answered.mean())
    risk = float((~correct[answered]).mean()) if answered.any() else float("nan")
    return SelectiveResult(
        answer_rate=answer_rate,
        abstention_rate=1.0 - answer_rate,
        risk=risk,
    )


def ranking_metrics(ranks: np.ndarray) -> RankingResult:
    values = np.asarray(ranks, dtype=np.int64).reshape(-1)
    if len(values) == 0 or np.any(values < 1):
        raise ValueError("ranks must be nonempty positive integers")
    return RankingResult(
        mrr=float((1.0 / values).mean()),
        hits_at_1=float((values <= 1).mean()),
        hits_at_3=float((values <= 3).mean()),
        hits_at_10=float((values <= 10).mean()),
    )
