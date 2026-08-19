from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from .config import ExperimentConfig


@dataclass(frozen=True)
class QuadrupleTable:
    values: np.ndarray
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    timestamp_to_id: dict[str, int]
    input_row_count: int | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != 4:
            raise ValueError("quadruple values must have shape (n, 4)")
        if not np.issubdtype(self.values.dtype, np.integer):
            raise ValueError("quadruple values must contain integer IDs")
        if self.input_row_count is not None and self.input_row_count < len(self.values):
            raise ValueError("input_row_count cannot be smaller than normalized facts")

    @property
    def timestamps(self) -> np.ndarray:
        return self.values[:, 3]


@dataclass(frozen=True)
class TemporalSplit:
    train: QuadrupleTable
    calibration: QuadrupleTable
    test: QuadrupleTable


@dataclass(frozen=True)
class CalibrationRoles:
    scorer_validation: QuadrupleTable
    calibrator_tuning: QuadrupleTable
    selector_validation: QuadrupleTable
    final_calibration: QuadrupleTable

    def as_tuple(self) -> tuple[QuadrupleTable, QuadrupleTable, QuadrupleTable, QuadrupleTable]:
        return (
            self.scorer_validation,
            self.calibrator_tuning,
            self.selector_validation,
            self.final_calibration,
        )


def table_fingerprint(
    table: QuadrupleTable,
    source_hashes: dict[str, str] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(table.values.tobytes())
    identity = {
        "entities": sorted(table.entity_to_id.items()),
        "relations": sorted(table.relation_to_id.items()),
        "timestamps": sorted(table.timestamp_to_id.items()),
        "source_hashes": dict(sorted((source_hashes or {}).items())),
    }
    digest.update(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def add_inverse_relations(values: np.ndarray, num_relations: int) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("values must have shape (n, 4)")
    if num_relations <= 0:
        raise ValueError("num_relations must be positive")
    inverse = values[:, [2, 1, 0, 3]].copy()
    inverse[:, 1] += num_relations
    return np.concatenate((values.copy(), inverse), axis=0)


def build_table(rows: Iterable[tuple[str, str, str, str]]) -> QuadrupleTable:
    input_rows = list(rows)
    def timestamp_key(value: str) -> tuple[int, Decimal | str]:
        try:
            return (0, Decimal(value))
        except InvalidOperation:
            return (1, value)

    unique_rows = sorted(
        set(input_rows),
        key=lambda row: (timestamp_key(row[3]), row[0], row[1], row[2]),
    )
    if not unique_rows:
        raise ValueError("rows must not be empty")
    entities = sorted({row[0] for row in unique_rows} | {row[2] for row in unique_rows})
    relations = sorted({row[1] for row in unique_rows})
    timestamps = sorted({row[3] for row in unique_rows}, key=timestamp_key)
    entity_to_id = {value: index for index, value in enumerate(entities)}
    relation_to_id = {value: index for index, value in enumerate(relations)}
    timestamp_to_id = {value: index for index, value in enumerate(timestamps)}
    values = np.asarray(
        [
            (
                entity_to_id[subject],
                relation_to_id[relation],
                entity_to_id[object_],
                timestamp_to_id[timestamp],
            )
            for subject, relation, object_, timestamp in unique_rows
        ],
        dtype=np.int64,
    )
    return QuadrupleTable(
        values,
        entity_to_id,
        relation_to_id,
        timestamp_to_id,
        input_row_count=len(input_rows),
    )


def _subset(table: QuadrupleTable, mask: np.ndarray) -> QuadrupleTable:
    return QuadrupleTable(
        table.values[mask].copy(),
        table.entity_to_id,
        table.relation_to_id,
        table.timestamp_to_id,
        input_row_count=int(mask.sum()),
    )


def temporal_split(
    table: QuadrupleTable,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> TemporalSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train and calibration fractions must leave a test partition")

    timestamps = np.unique(table.timestamps)
    train_count = int(np.floor(train_fraction * len(timestamps)))
    calibration_end = int(
        np.floor((train_fraction + calibration_fraction) * len(timestamps))
    )
    if train_count < 1 or calibration_end <= train_count or calibration_end >= len(timestamps):
        raise ValueError("temporal split must leave at least one timestamp per partition")

    train_max = timestamps[train_count - 1]
    calibration_max = timestamps[calibration_end - 1]
    return TemporalSplit(
        train=_subset(table, table.timestamps <= train_max),
        calibration=_subset(
            table,
            (table.timestamps > train_max) & (table.timestamps <= calibration_max),
        ),
        test=_subset(table, table.timestamps > calibration_max),
    )


def split_model_selection(
    calibration: QuadrupleTable,
    fraction: float = 0.25,
) -> tuple[QuadrupleTable, QuadrupleTable]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("model-selection fraction must be between 0 and 1")
    timestamps = np.unique(calibration.timestamps)
    selection_count = max(1, int(np.floor(fraction * len(timestamps))))
    if selection_count >= len(timestamps):
        raise ValueError("calibration must contain at least two timestamps")
    boundary = timestamps[selection_count - 1]
    return (
        _subset(calibration, calibration.timestamps <= boundary),
        _subset(calibration, calibration.timestamps > boundary),
    )


def split_calibration_roles(
    calibration: QuadrupleTable,
    fractions: Sequence[float] = (0.25, 0.30, 0.15, 0.30),
) -> CalibrationRoles:
    if len(fractions) != 4:
        raise ValueError("calibration role split requires four fractions")
    if any(value <= 0.0 for value in fractions):
        raise ValueError("calibration role fractions must be positive")
    total = float(sum(fractions))
    if not np.isclose(total, 1.0):
        raise ValueError("calibration role fractions must sum to 1")

    timestamps = np.unique(calibration.timestamps)
    if len(timestamps) < 4:
        raise ValueError("calibration role split requires at least four timestamps")

    raw_counts = np.asarray(fractions, dtype=float) * len(timestamps)
    counts = np.floor(raw_counts).astype(int)
    counts[counts == 0] = 1
    while counts.sum() > len(timestamps):
        candidates = np.where(counts > 1)[0]
        if len(candidates) == 0:
            raise ValueError("cannot allocate nonempty calibration role splits")
        index = int(candidates[np.argmax(counts[candidates])])
        counts[index] -= 1
    remainders = raw_counts - np.floor(raw_counts)
    while counts.sum() < len(timestamps):
        for index in np.argsort(-remainders):
            if counts.sum() >= len(timestamps):
                break
            counts[int(index)] += 1
            remainders[int(index)] = -1.0

    boundaries = np.cumsum(counts)
    partitions = np.split(timestamps, boundaries[:-1])
    if any(len(partition) == 0 for partition in partitions):
        raise ValueError("calibration role split produced an empty partition")

    return CalibrationRoles(
        scorer_validation=_subset(
            calibration,
            np.isin(calibration.timestamps, partitions[0]),
        ),
        calibrator_tuning=_subset(
            calibration,
            np.isin(calibration.timestamps, partitions[1]),
        ),
        selector_validation=_subset(
            calibration,
            np.isin(calibration.timestamps, partitions[2]),
        ),
        final_calibration=_subset(
            calibration,
            np.isin(calibration.timestamps, partitions[3]),
        ),
    )


def load_quadruple_files(paths: Iterable[Path]) -> QuadrupleTable:
    rows: list[tuple[str, str, str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected four columns")
            rows.append((fields[0], fields[1], fields[2], fields[3]))
    return build_table(rows)


def built_in_toy_table() -> QuadrupleTable:
    entities = ("alice", "bob", "carol", "dave")
    rows: list[tuple[str, str, str, str]] = []
    for day in range(15):
        subject = entities[day % len(entities)]
        rows.append(
            (
                subject,
                "supports",
                entities[(day + 1) % len(entities)],
                f"2020-01-{day + 1:02d}",
            )
        )
        rows.append(
            (
                subject,
                "visits",
                entities[(day + 2) % len(entities)],
                f"2020-01-{day + 1:02d}",
            )
        )
    return build_table(rows)


def load_configured_table(config: "ExperimentConfig") -> QuadrupleTable:
    if config.data_mode == "toy":
        return built_in_toy_table()
    paths = [config.data_path / name for name in ("train.txt", "valid.txt", "test.txt")]
    return load_quadruple_files(paths)
