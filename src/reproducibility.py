from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set deterministic random seeds for supported libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: str | Path, output_path: str | Path,
                   include_suffixes: Iterable[str] = (".py", ".yaml", ".md", ".csv", ".json")) -> None:
    root_path = Path(root).resolve()
    output = Path(output_path).resolve()
    rows = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        if path.suffix.lower() not in include_suffixes:
            continue
        rows.append({
            "path": str(path.relative_to(root_path)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
