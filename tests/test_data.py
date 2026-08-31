import numpy as np

from riskcal_tkg.config import ExperimentConfig

from riskcal_tkg.data import (
    add_inverse_relations,
    build_table,
    configured_quadruple_paths,
    load_configured_table,
    split_calibration_roles,
    split_model_selection,
    table_fingerprint,
    temporal_split,
)


ROWS = [
    ("b", "r", "c", "2020-01-03"),
    ("a", "r", "b", "2020-01-01"),
    ("a", "s", "c", "2020-01-02"),
    ("c", "r", "a", "2020-01-04"),
    ("b", "s", "a", "2020-01-05"),
]


def test_mapping_is_independent_of_input_order() -> None:
    forward = build_table(ROWS)
    reverse = build_table(list(reversed(ROWS)))
    assert forward.entity_to_id == reverse.entity_to_id
    assert forward.relation_to_id == reverse.relation_to_id
    assert np.array_equal(forward.values, reverse.values)


def test_temporal_split_never_splits_a_timestamp() -> None:
    table = build_table(ROWS)
    split = temporal_split(table, train_fraction=0.6, calibration_fraction=0.2)
    assert max(split.train.timestamps) < min(split.calibration.timestamps)
    assert max(split.calibration.timestamps) < min(split.test.timestamps)
    assert set(split.train.timestamps).isdisjoint(split.calibration.timestamps)


def test_build_table_deduplicates_exact_facts() -> None:
    table = build_table([ROWS[0], ROWS[0], ROWS[1]])
    assert len(table.values) == 2
    assert table.input_row_count == 3


def test_numeric_timestamps_are_sorted_by_time_not_lexicographically() -> None:
    rows = [
        ("a", "r", "b", "10"),
        ("a", "r", "b", "2"),
        ("a", "r", "b", "1"),
        ("a", "r", "b", "365"),
    ]
    table = build_table(rows)
    ordered = sorted(table.timestamp_to_id, key=table.timestamp_to_id.get)
    assert ordered == ["1", "2", "10", "365"]


def test_inverse_relations_turn_subject_prediction_into_object_prediction() -> None:
    values = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    augmented = add_inverse_relations(values, num_relations=4)
    assert augmented.tolist() == [[0, 1, 2, 3], [2, 5, 0, 3]]


def test_model_selection_uses_only_earliest_calibration_timestamps() -> None:
    table = build_table(ROWS)
    selection, conformal = split_model_selection(table, fraction=0.25)
    assert max(selection.timestamps) < min(conformal.timestamps)
    assert len(np.unique(selection.timestamps)) == 1


def test_calibration_roles_are_contiguous_and_non_overlapping() -> None:
    rows = [
        ("a", "r", "b", f"2020-02-{day + 1:02d}")
        for day in range(12)
    ]
    table = build_table(rows)
    roles = split_calibration_roles(table, fractions=(0.25, 0.25, 0.25, 0.25))
    role_timestamps = [np.unique(role.timestamps) for role in roles.as_tuple()]
    assert [len(values) for values in role_timestamps] == [3, 3, 3, 3]
    assert max(role_timestamps[0]) < min(role_timestamps[1])
    assert max(role_timestamps[1]) < min(role_timestamps[2])
    assert max(role_timestamps[2]) < min(role_timestamps[3])
    assert len(set(np.concatenate(role_timestamps))) == 12


def test_dataset_fingerprint_includes_semantic_id_mappings() -> None:
    first = build_table([("alice", "supports", "bob", "1")])
    renamed = build_table([("x", "related", "y", "1")])
    assert np.array_equal(first.values, renamed.values)
    assert table_fingerprint(first) != table_fingerprint(renamed)


def test_configured_dataset_accepts_official_bare_split_names(tmp_path) -> None:
    for name, row in zip(("train", "valid", "test"), ROWS[:3], strict=True):
        (tmp_path / name).write_text("\t".join(row) + "\n", encoding="utf-8")
    config = ExperimentConfig(data_mode="icews05_15", data_path=tmp_path)
    assert [path.name for path in configured_quadruple_paths(config)] == [
        "train",
        "valid",
        "test",
    ]
    table = load_configured_table(config)
    assert len(table.values) == 3
