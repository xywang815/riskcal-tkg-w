from pathlib import Path

import pytest

from riskcal_tkg.config import ExperimentConfig, load_config


def test_load_config_returns_validated_values(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "seed: 17\ndevice: cpu\ntarget_coverage: 0.9\n"
        "deletion_rates: [0.0, 0.1]\nembedding_dim: 16\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config == ExperimentConfig(
        seed=17,
        device="cpu",
        target_coverage=0.9,
        deletion_rates=(0.0, 0.1),
        embedding_dim=16,
    )


def test_config_rejects_invalid_coverage(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("target_coverage: 1.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target_coverage"):
        load_config(path)


def test_load_config_accepts_training_parameters(tmp_path: Path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text(
        "epochs: 20\nbatch_size: 64\nnegatives: 8\n"
        "training_margin: 0.5\nlearning_rate: 0.01\n"
        "weight_decay: 0.0001\nmin_score_std: 1e-9\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.epochs == 20
    assert config.batch_size == 64
    assert config.negatives == 8
    assert config.training_margin == 0.5
    assert config.learning_rate == 0.01
    assert config.weight_decay == 0.0001
    assert config.min_score_std == 1e-9


def test_load_config_accepts_expansion_parameters(tmp_path: Path) -> None:
    path = tmp_path / "expansion.yaml"
    path.write_text(
        "data_mode: icews05_15\n"
        "model_name: continuous_tcomplex\n"
        "time_encoding: bounded_fourier\n"
        "negative_sampling: filtered\n"
        "include_kgcp_baselines: true\n"
        "explicit_method_names: true\n"
        "kgcp_temperature: 1.0\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.data_mode == "icews05_15"
    assert config.model_name == "continuous_tcomplex"
    assert config.time_encoding == "bounded_fourier"
    assert config.negative_sampling == "filtered"
    assert config.include_kgcp_baselines
    assert config.explicit_method_names
    assert config.kgcp_temperature == 1.0


def test_config_rejects_unknown_time_encoding(tmp_path: Path) -> None:
    path = tmp_path / "bad_time.yaml"
    path.write_text("time_encoding: future_oracle\n", encoding="utf-8")
    with pytest.raises(ValueError, match="time_encoding"):
        load_config(path)


def test_load_config_accepts_preregistered_selection_parameters(tmp_path: Path) -> None:
    path = tmp_path / "selection.yaml"
    path.write_text(
        "half_lives: [7, 14, 30, .inf]\n"
        "calibration_role_fractions: [0.25, 0.30, 0.15, 0.30]\n"
        "adaptive_selector_ridge: 0.5\n"
        "adaptive_coverage_tolerance: 0.03\n"
        "eval_every: 5\npatience: 10\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.half_lives[:3] == (7.0, 14.0, 30.0)
    assert config.half_lives[3] == float("inf")
    assert config.calibration_role_fractions == (0.25, 0.30, 0.15, 0.30)
    assert config.adaptive_selector_ridge == 0.5
    assert config.adaptive_coverage_tolerance == 0.03
    assert config.eval_every == 5
    assert config.patience == 10
