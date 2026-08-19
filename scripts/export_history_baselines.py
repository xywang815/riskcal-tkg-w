"""Export deterministic history-based ranking baselines for ICEWS14.

The script intentionally has few dependencies. It recomputes the project ID
mapping, temporal split, and deletion masks from raw quadruple files, then
validates each recomputed mask against the archived confirmatory run when
deletion-mask audits are available.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_SEEDS = (17, 29, 43, 59, 71)
DEFAULT_DELETION_RATES = (0.0, 0.1, 0.2, 0.3)


@dataclass(frozen=True)
class QuadrupleTable:
    values: np.ndarray
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    timestamp_to_id: dict[str, int]


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g", lineterminator="\n")
    temporary.replace(path)


def _timestamp_key(value: str) -> tuple[int, Decimal | str]:
    try:
        return (0, Decimal(value))
    except InvalidOperation:
        return (1, value)


def _build_table(paths: Iterable[Path]) -> QuadrupleTable:
    rows: list[tuple[str, str, str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected four columns")
            rows.append((fields[0], fields[1], fields[2], fields[3]))
    unique_rows = sorted(
        set(rows),
        key=lambda row: (_timestamp_key(row[3]), row[0], row[1], row[2]),
    )
    entities = sorted({row[0] for row in unique_rows} | {row[2] for row in unique_rows})
    relations = sorted({row[1] for row in unique_rows})
    timestamps = sorted({row[3] for row in unique_rows}, key=_timestamp_key)
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
    return QuadrupleTable(values, entity_to_id, relation_to_id, timestamp_to_id)


def _temporal_split(
    values: np.ndarray,
    *,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.unique(values[:, 3])
    train_count = int(np.floor(train_fraction * len(timestamps)))
    calibration_end = int(np.floor((train_fraction + calibration_fraction) * len(timestamps)))
    train_max = timestamps[train_count - 1]
    calibration_max = timestamps[calibration_end - 1]
    return (
        values[values[:, 3] <= train_max].copy(),
        values[(values[:, 3] > train_max) & (values[:, 3] <= calibration_max)].copy(),
        values[values[:, 3] > calibration_max].copy(),
    )


def _add_inverse(values: np.ndarray, relation_count: int) -> np.ndarray:
    inverse = values[:, [2, 1, 0, 3]].copy()
    inverse[:, 1] += relation_count
    return np.concatenate((values.copy(), inverse), axis=0)


def _delete_training_edges(values: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, str]:
    keep = np.ones(len(values), dtype=bool)
    target = int(np.floor(rate * len(values)))
    if len(values):
        protected = np.zeros(len(values), dtype=bool)
        for entity_id in np.unique(values[:, [0, 2]]):
            incident = np.flatnonzero((values[:, 0] == entity_id) | (values[:, 2] == entity_id))
            earliest_timestamp = values[incident, 3].min()
            earliest = incident[values[incident, 3] == earliest_timestamp]
            protected[int(earliest.min())] = True
        deleted = 0
        for index in np.random.default_rng(seed).permutation(len(values)):
            if deleted >= target:
                break
            if protected[index]:
                continue
            keep[index] = False
            deleted += 1
    digest = hashlib.sha256(np.packbits(keep).tobytes()).hexdigest()
    return values[keep].copy(), digest


def _truth_index(values: np.ndarray) -> dict[tuple[int, int, int], set[int]]:
    index: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for subject, relation, object_, timestamp in values:
        index[(int(subject), int(relation), int(timestamp))].add(int(object_))
    return dict(index)


def _ranking_metrics(ranks: list[int]) -> dict[str, float]:
    values = np.asarray(ranks, dtype=np.int64)
    return {
        "mrr": float((1.0 / values).mean()),
        "hits_at_1": float((values <= 1).mean()),
        "hits_at_3": float((values <= 3).mean()),
        "hits_at_10": float((values <= 10).mean()),
    }


class HistoryRanker:
    def __init__(self, entity_count: int, history: np.ndarray) -> None:
        self.entity_count = entity_count
        self.entity_ids = np.arange(entity_count, dtype=np.int64)
        self.global_counts = np.zeros(entity_count, dtype=np.int64)
        self.rel_counts: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(entity_count, dtype=np.int64)
        )
        self.pair_counts: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        self._frequency_positions: np.ndarray | None = None
        self._secondary_positions: dict[int, np.ndarray] = {}
        self.add(history)

    def add(self, facts: np.ndarray) -> None:
        for subject, relation, object_, _ in facts:
            subject = int(subject)
            relation = int(relation)
            object_ = int(object_)
            self.global_counts[object_] += 1
            self.rel_counts[relation][object_] += 1
            key = (subject, relation)
            self.pair_counts[key][object_] = self.pair_counts[key].get(object_, 0) + 1
        self._frequency_positions = None
        self._secondary_positions = {}

    def _positions(self, primary: np.ndarray, secondary: np.ndarray | None = None) -> np.ndarray:
        if secondary is None:
            secondary = np.zeros(self.entity_count, dtype=np.int64)
        order = np.lexsort((self.entity_ids, -secondary, -primary))
        positions = np.empty(self.entity_count, dtype=np.int64)
        positions[order] = np.arange(self.entity_count)
        return positions

    def _frequency_rank_positions(self) -> np.ndarray:
        if self._frequency_positions is None:
            self._frequency_positions = self._positions(self.global_counts)
        return self._frequency_positions

    def _relation_rank_positions(self, relation: int) -> np.ndarray:
        if relation not in self._secondary_positions:
            self._secondary_positions[relation] = self._positions(
                self.rel_counts[relation], self.global_counts
            )
        return self._secondary_positions[relation]

    def _secondary_key(self, relation: int, entity: int) -> tuple[int, int, int]:
        return (
            int(self.rel_counts[relation][entity]),
            int(self.global_counts[entity]),
            -entity,
        )

    @staticmethod
    def _filtered_position_rank(
        positions: np.ndarray,
        true_entity: int,
        other_true: set[int],
    ) -> int:
        true_position = int(positions[true_entity])
        ahead_other = sum(
            1
            for entity in other_true
            if entity != true_entity and int(positions[int(entity)]) < true_position
        )
        return true_position + 1 - ahead_other

    def frequency_rank(self, true_entity: int, other_true: set[int]) -> int:
        return self._filtered_position_rank(
            self._frequency_rank_positions(), true_entity, other_true
        )

    def relation_frequency_rank(
        self,
        relation: int,
        true_entity: int,
        other_true: set[int],
    ) -> int:
        return self._filtered_position_rank(
            self._relation_rank_positions(relation), true_entity, other_true
        )

    def repeat_rank(
        self,
        subject: int,
        relation: int,
        true_entity: int,
        other_true: set[int],
    ) -> int:
        pair_counts = self.pair_counts.get((subject, relation), {})
        true_pair_count = pair_counts.get(true_entity, 0)
        true_secondary = self._secondary_key(relation, true_entity)
        if true_pair_count > 0:
            ahead = 0
            for entity, count in pair_counts.items():
                if entity == true_entity:
                    continue
                if count > true_pair_count or (
                    count == true_pair_count
                    and self._secondary_key(relation, entity) > true_secondary
                ):
                    ahead += 1
        else:
            positions = self._relation_rank_positions(relation)
            ahead = int(positions[true_entity])
            for entity, count in pair_counts.items():
                if count <= 0:
                    continue
                if int(positions[int(entity)]) < int(positions[true_entity]):
                    ahead -= 1
                ahead += 1

        true_key = (true_pair_count, *true_secondary)
        ahead_other = 0
        for entity in other_true:
            entity = int(entity)
            if entity == true_entity:
                continue
            candidate_key = (
                pair_counts.get(entity, 0),
                *self._secondary_key(relation, entity),
            )
            if candidate_key > true_key:
                ahead_other += 1
        return 1 + ahead - ahead_other


def _evaluate_history(
    test_facts: np.ndarray,
    history: np.ndarray,
    truth: dict[tuple[int, int, int], set[int]],
    *,
    entity_count: int,
    prequential: bool,
) -> dict[str, dict[str, float]]:
    ranker = HistoryRanker(entity_count, history)
    ranks: dict[str, list[int]] = {
        "frequency": [],
        "relation_frequency": [],
        "repeat": [],
    }
    for timestamp in np.unique(test_facts[:, 3]):
        batch = test_facts[test_facts[:, 3] == timestamp]
        for subject, relation, object_, _ in batch:
            subject = int(subject)
            relation = int(relation)
            object_ = int(object_)
            timestamp = int(timestamp)
            other_true = truth[(subject, relation, timestamp)] - {object_}
            ranks["frequency"].append(ranker.frequency_rank(object_, other_true))
            ranks["relation_frequency"].append(
                ranker.relation_frequency_rank(relation, object_, other_true)
            )
            ranks["repeat"].append(
                ranker.repeat_rank(subject, relation, object_, other_true)
            )
        if prequential:
            ranker.add(batch)
    return {name: _ranking_metrics(values) for name, values in ranks.items()}


def _rate_label(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def export_history_baselines(
    data_root: Path,
    run_root: Path,
    paper_root: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    deletion_rates: tuple[float, ...] = DEFAULT_DELETION_RATES,
) -> dict[str, Any]:
    table = _build_table(data_root / name for name in ("train.txt", "valid.txt", "test.txt"))
    train, calibration, test = _temporal_split(table.values)
    relation_count = len(table.relation_to_id)
    entity_count = len(table.entity_to_id)
    truth = _truth_index(_add_inverse(table.values, relation_count))
    calibration_inverse = _add_inverse(calibration, relation_count)
    test_inverse = _add_inverse(test, relation_count)

    records: list[dict[str, Any]] = []
    validated_masks = 0
    for seed in seeds:
        for rate in deletion_rates:
            retained_train, mask_sha256 = _delete_training_edges(train, rate, seed)
            audit_path = run_root / "deletion_masks" / f"seed{seed}_delete{_rate_label(rate)}.json"
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if mask_sha256 != audit["mask_sha256"]:
                    raise ValueError(f"deletion mask mismatch for {audit_path.name}")
                validated_masks += 1
            train_history = _add_inverse(retained_train, relation_count)
            histories = {
                "train_only": (train_history, False),
                "prequential": (
                    np.concatenate((train_history, calibration_inverse), axis=0),
                    True,
                ),
            }
            for history_mode, (history, prequential) in histories.items():
                metrics = _evaluate_history(
                    test_inverse,
                    history,
                    truth,
                    entity_count=entity_count,
                    prequential=prequential,
                )
                for baseline, values in metrics.items():
                    records.append(
                        {
                            "seed": seed,
                            "deletion_rate": rate,
                            "history_mode": history_mode,
                            "baseline": baseline,
                            "query_count": len(test_inverse),
                            **values,
                        }
                    )

    by_seed = pd.DataFrame(records)
    summary = (
        by_seed.groupby(["deletion_rate", "history_mode", "baseline"], as_index=False)
        .agg(
            mrr=("mrr", "mean"),
            hits_at_1=("hits_at_1", "mean"),
            hits_at_3=("hits_at_3", "mean"),
            hits_at_10=("hits_at_10", "mean"),
            query_count=("query_count", "first"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["deletion_rate", "history_mode", "baseline"])
        .reset_index(drop=True)
    )
    data_dir = paper_root / "data" / "final_confirmatory"
    _write_csv(by_seed, data_dir / "history_baseline_by_seed.csv")
    _write_csv(summary, data_dir / "history_baseline_summary.csv")
    manifest = {
        "baselines": ["frequency", "relation_frequency", "repeat"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "deletion_rates": list(deletion_rates),
        "history_modes": {
            "train_only": "retained training facts only",
            "prequential": (
                "retained training plus calibration facts before the first test "
                "timestamp, then test facts are added only after each complete "
                "timestamp batch is predicted"
            ),
        },
        "mask_audits_validated": validated_masks,
        "run_root": str(run_root),
        "seeds": list(seeds),
    }
    _write_json(manifest, data_dir / "history_baseline_manifest.json")
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/icews14"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--deletion-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELETION_RATES),
    )
    args = parser.parse_args()
    manifest = export_history_baselines(
        args.data_root,
        args.run_root,
        args.paper_root,
        seeds=tuple(args.seeds),
        deletion_rates=tuple(args.deletion_rates),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
