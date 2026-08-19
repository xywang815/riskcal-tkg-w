from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 17
    seeds: tuple[int, ...] = ()
    device: str = "auto"
    data_mode: str = "toy"
    data_path: Path = Path("data/raw/icews14")
    output_root: Path = Path("results")
    target_coverage: float = 0.9
    deletion_rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    train_fraction: float = 0.6
    calibration_fraction: float = 0.2
    calibration_role_fractions: tuple[float, float, float, float] = (
        0.25,
        0.30,
        0.15,
        0.30,
    )
    rolling_window: int = 1000
    half_lives: tuple[float, ...] = (7.0, 14.0, 30.0, math.inf)
    adaptive_selector_ridge: float = 1.0
    adaptive_coverage_tolerance: float = 0.02
    max_set_sizes: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
    embedding_dim: int = 128
    epochs: int = 100
    batch_size: int = 512
    negatives: int = 64
    training_margin: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    eval_every: int = 5
    patience: int = 10
    min_score_std: float = 1e-10

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.data_mode not in {"toy", "icews14"}:
            raise ValueError("data_mode must be toy or icews14")
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError("target_coverage must be between 0 and 1")
        if any(rate < 0.0 or rate >= 1.0 for rate in self.deletion_rates):
            raise ValueError("deletion_rates must be in [0, 1)")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be between 0 and 1")
        if self.train_fraction + self.calibration_fraction >= 1.0:
            raise ValueError("split fractions must leave a test partition")
        if len(self.calibration_role_fractions) != 4:
            raise ValueError("calibration_role_fractions must contain four values")
        if any(value <= 0.0 for value in self.calibration_role_fractions):
            raise ValueError("calibration_role_fractions must be positive")
        if not math.isclose(sum(self.calibration_role_fractions), 1.0):
            raise ValueError("calibration_role_fractions must sum to 1")
        if self.rolling_window <= 0 or not self.half_lives:
            raise ValueError("calibration window and half-life candidates must be nonempty")
        if any(value <= 0 or math.isnan(value) for value in self.half_lives):
            raise ValueError("half-life candidates must be positive")
        if self.adaptive_selector_ridge < 0:
            raise ValueError("adaptive_selector_ridge must be nonnegative")
        if not 0.0 <= self.adaptive_coverage_tolerance < 1.0:
            raise ValueError("adaptive_coverage_tolerance must be in [0, 1)")
        if any(size < 0 for size in self.max_set_sizes):
            raise ValueError("max_set_sizes must be nonnegative")
        if min(
            self.epochs,
            self.batch_size,
            self.negatives,
            self.eval_every,
            self.patience,
        ) <= 0:
            raise ValueError("training sizes must be positive")
        if self.training_margin < 0.0:
            raise ValueError("training_margin must be nonnegative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")
        if self.min_score_std < 0.0:
            raise ValueError("min_score_std must be nonnegative")


def load_config(path: Path) -> ExperimentConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    if "deletion_rates" in raw:
        raw["deletion_rates"] = tuple(float(value) for value in raw["deletion_rates"])
    if "seeds" in raw:
        raw["seeds"] = tuple(int(value) for value in raw["seeds"])
    if "max_set_sizes" in raw:
        raw["max_set_sizes"] = tuple(int(value) for value in raw["max_set_sizes"])
    if "half_lives" in raw:
        raw["half_lives"] = tuple(float(value) for value in raw["half_lives"])
    if "calibration_role_fractions" in raw:
        raw["calibration_role_fractions"] = tuple(
            float(value) for value in raw["calibration_role_fractions"]
        )
    for key in (
        "target_coverage",
        "train_fraction",
        "calibration_fraction",
        "adaptive_selector_ridge",
        "adaptive_coverage_tolerance",
        "training_margin",
        "learning_rate",
        "weight_decay",
        "min_score_std",
    ):
        if key in raw:
            raw[key] = float(raw[key])
    for key in ("data_path", "output_root"):
        if key in raw:
            raw[key] = Path(raw[key])
    return ExperimentConfig(**raw)
