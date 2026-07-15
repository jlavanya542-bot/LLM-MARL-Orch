from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


_WEIGHT_KEYS = ("sla", "energy", "imbalance", "fairness", "carbon")


class SemanticAgent:
    """Context-to-objective mapper with deterministic caching and optional local inference."""

    def __init__(self, cfg: Dict, cache_path: str | Path | None = None):
        self.cfg = cfg
        self.backend = str(cfg["semantic_agent"]["backend"])
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Dict[str, float]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    @staticmethod
    def build_prompt(workload: pd.Series) -> str:
        return (
            "Return normalized scheduling priorities for SLA, energy, imbalance, fairness, and carbon. "
            f"Workload class={workload['workload_class']}; priority={workload['priority']}; "
            f"latency_budget_ms={float(workload['latency_budget_ms']):.2f}; "
            f"energy_sensitivity={float(workload['energy_sensitivity']):.3f}; "
            f"edge_locality={float(workload['edge_locality']):.3f}; "
            f"regime={workload['stress_regime']}; description={workload['semantic_description']}"
        )

    def _cache_key(self, workload: pd.Series) -> str:
        return "|".join([
            str(workload["workload_class"]),
            str(int(workload["priority"])),
            str(workload["stress_regime"]),
            f"{float(workload['edge_locality']):.1f}",
            f"{float(workload['energy_sensitivity']):.1f}",
        ])

    def _deterministic_weights(self, workload: pd.Series) -> Dict[str, float]:
        base = dict(self.cfg["semantic_agent"]["weights"][str(workload["workload_class"])])
        if int(workload["priority"]) == 3:
            base["sla"] += 0.06
            base["energy"] -= 0.03
            base["carbon"] -= 0.03
        if str(workload["stress_regime"]) == "energy-scarcity":
            base["energy"] += 0.05
            base["carbon"] += 0.03
            base["sla"] -= 0.05
            base["imbalance"] -= 0.03
        if float(workload["edge_locality"]) > 0.80:
            base["sla"] += 0.03
            base["fairness"] -= 0.02
            base["imbalance"] -= 0.01
        values = np.array([base[k] for k in _WEIGHT_KEYS], dtype=float)
        values = np.clip(values, 0.01, None)
        values /= values.sum()
        return {k: float(v) for k, v in zip(_WEIGHT_KEYS, values)}

    def _local_transformer_weights(self, workload: pd.Series) -> Dict[str, float]:
        # The local backend is intentionally optional. It never downloads a model silently.
        try:
            from transformers import pipeline
            model_name = str(self.cfg["semantic_agent"]["local_model_name"])
            generator = pipeline("text2text-generation", model=model_name, local_files_only=True)
            output = generator(self.build_prompt(workload), max_new_tokens=64)[0]["generated_text"]
            numbers = [float(x) for x in output.replace(",", " ").split() if x.replace(".", "", 1).isdigit()]
            if len(numbers) >= 5:
                values = np.clip(np.array(numbers[:5], dtype=float), 0.01, None)
                values /= values.sum()
                return {k: float(v) for k, v in zip(_WEIGHT_KEYS, values)}
        except Exception:
            pass
        return self._deterministic_weights(workload)

    def infer(self, workload: pd.Series, force_refresh: bool = False) -> Tuple[Dict[str, float], float, bool]:
        key = self._cache_key(workload)
        if not force_refresh and key in self.cache:
            return dict(self.cache[key]), 0.0, True

        start = time.perf_counter()
        if self.backend == "local_transformer":
            weights = self._local_transformer_weights(workload)
        else:
            weights = self._deterministic_weights(workload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.cache[key] = weights
        return weights, float(latency_ms), False

    def persist_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")


class SemanticDriftDetector:
    def __init__(self, classes, window: int = 100, threshold: float = 0.18):
        self.classes = list(classes)
        self.window = int(window)
        self.threshold = float(threshold)
        self.history = deque(maxlen=self.window)
        self.reference = np.ones(len(self.classes), dtype=float) / len(self.classes)

    def fit_reference(self, workload_classes) -> None:
        counts = Counter(map(str, workload_classes))
        values = np.array([counts.get(c, 0) + 1e-6 for c in self.classes], dtype=float)
        self.reference = values / values.sum()

    def update(self, workload_class: str) -> Tuple[float, bool]:
        self.history.append(str(workload_class))
        if len(self.history) < max(10, self.window // 4):
            return 0.0, False
        counts = Counter(self.history)
        observed = np.array([counts.get(c, 0) + 1e-6 for c in self.classes], dtype=float)
        observed /= observed.sum()
        divergence = float(np.sum(observed * np.log(observed / self.reference)))
        return divergence, bool(divergence > self.threshold)
