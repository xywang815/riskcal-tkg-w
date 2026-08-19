"""Risk-calibrated temporal knowledge graph experiments."""

__all__ = ["ExperimentConfig", "load_config"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from .config import ExperimentConfig, load_config

        exports = {
            "ExperimentConfig": ExperimentConfig,
            "load_config": load_config,
        }
        return exports[name]
    raise AttributeError(name)
