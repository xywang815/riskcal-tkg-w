from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class DeletionResult:
    values: np.ndarray
    keep_mask: np.ndarray
    requested_rate: float
    actual_rate: float
    mask_sha256: str


def delete_training_edges(values: np.ndarray, rate: float, seed: int) -> DeletionResult:
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("values must have shape (n, 4)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("values must contain integer IDs")

    keep = np.ones(len(values), dtype=bool)
    target = int(np.floor(rate * len(values)))
    if len(values):
        protected = np.zeros(len(values), dtype=bool)
        for entity_id in np.unique(values[:, [0, 2]]):
            incident = np.flatnonzero(
                (values[:, 0] == entity_id) | (values[:, 2] == entity_id)
            )
            earliest_timestamp = values[incident, 3].min()
            earliest = incident[values[incident, 3] == earliest_timestamp]
            protected[int(earliest.min())] = True
        order = np.random.default_rng(seed).permutation(len(values))
        deleted = 0
        for index in order:
            if deleted >= target:
                break
            if protected[index]:
                continue
            keep[index] = False
            deleted += 1

    digest = hashlib.sha256(np.packbits(keep).tobytes()).hexdigest()
    actual_rate = 0.0 if len(values) == 0 else float((~keep).sum() / len(values))
    return DeletionResult(
        values=values[keep].copy(),
        keep_mask=keep,
        requested_rate=rate,
        actual_rate=actual_rate,
        mask_sha256=digest,
    )
