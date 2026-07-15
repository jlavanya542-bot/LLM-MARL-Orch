from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate the YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    required = ["project", "reproducibility", "dataset", "simulation",
                "semantic_agent", "marl", "experiment", "outputs"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    fractions = (
        float(cfg["dataset"]["train_fraction"])
        + float(cfg["dataset"]["validation_fraction"])
        + float(cfg["dataset"]["test_fraction"])
    )
    if abs(fractions - 1.0) > 1e-8:
        raise ValueError("Dataset split fractions must sum to 1.0")

    if int(cfg["dataset"]["node_count"]) <= 0:
        raise ValueError("node_count must be positive")
    if int(cfg["dataset"]["workload_count"]) <= 0:
        raise ValueError("workload_count must be positive")

    return cfg
