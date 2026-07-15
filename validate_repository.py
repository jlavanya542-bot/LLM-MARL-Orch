from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.workload_generator import generate_nodes, generate_workloads


REQUIRED = [
    "README.md",
    "config.yaml",
    "requirements.txt",
    "run_reproducibility.py",
    "DATASET_CARD.md",
    "REPRODUCIBILITY.md",
    "CODE_AVAILABILITY.md",
    "DATA_AVAILABILITY.md",
    "src/workload_generator.py",
    "src/cloud_edge_environment.py",
    "src/semantic_agent.py",
    "src/marl_agent.py",
    "src/schedulers.py",
    "src/evaluator.py",
    "src/statistical_analysis.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    cfg = load_config(root / "config.yaml")
    nodes = generate_nodes(cfg, 7)
    workloads = generate_workloads(cfg, 7, 50)
    assert len(nodes) == int(cfg["dataset"]["node_count"])
    assert len(workloads) == 50
    assert set(workloads["workload_class"]).issubset(set(cfg["dataset"]["workload_classes"]))

    secret_patterns = [
        re.compile(r"(?i)(password|passwd|api[_-]?key)\s*[:=]\s*['\"][^'\"]+['\"]"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
    ]
    flagged = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".json", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in secret_patterns:
                if pattern.search(text):
                    flagged.append(str(path.relative_to(root)))
    if flagged:
        raise RuntimeError(f"Potential secrets detected in: {flagged}")

    print("Repository structure, configuration, dataset generator, and secret scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
