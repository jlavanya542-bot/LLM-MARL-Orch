from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .cloud_edge_environment import CloudEdgeEnvironment
from .marl_agent import CooperativeLinearQAgent
from .semantic_agent import SemanticAgent, SemanticDriftDetector


DEFAULT_WEIGHTS = {
    "sla": 0.34,
    "energy": 0.22,
    "imbalance": 0.20,
    "fairness": 0.12,
    "carbon": 0.12,
}


class BaseScheduler:
    name = "Base"

    def __init__(self, cfg: Dict, node_count: int, seed: int):
        self.cfg = cfg
        self.node_count = node_count
        self.seed = seed
        self.training = True
        self.current_weights = dict(DEFAULT_WEIGHTS)
        self.last_semantic_latency_ms = 0.0
        self.last_cache_hit = False
        self.last_drift = 0.0
        self.last_drift_triggered = False
        self._last_features = None
        self._last_action = None

    def set_training(self, value: bool) -> None:
        self.training = bool(value)

    def prepare(self, train_workloads: pd.DataFrame) -> None:
        pass

    def choose(self, env: CloudEdgeEnvironment, workload: pd.Series) -> int:
        raise NotImplementedError

    def observe(self, env: CloudEdgeEnvironment, workload: pd.Series,
                reward: float, next_workload: Optional[pd.Series]) -> None:
        pass

    def persist(self) -> None:
        pass


class K8sDefaultScheduler(BaseScheduler):
    name = "K8s-Default"

    def choose(self, env: CloudEdgeEnvironment, workload: pd.Series) -> int:
        f = env.candidate_features(workload)
        feasible = env.feasible_mask(workload)
        score = 0.55 * f[:, 3] + 0.30 * f[:, 2] + 0.15 * f[:, 8]
        score[~feasible] = np.inf
        if not np.any(feasible):
            score = 0.55 * f[:, 3] + 0.30 * f[:, 2] + 0.15 * f[:, 8]
        return int(np.argmin(score))


class EnergyAwareScheduler(BaseScheduler):
    name = "Energy-Aware"

    def choose(self, env: CloudEdgeEnvironment, workload: pd.Series) -> int:
        f = env.candidate_features(workload)
        feasible = env.feasible_mask(workload)
        score = 0.58 * f[:, 4] + 0.27 * f[:, 5] + 0.10 * f[:, 3] + 0.05 * f[:, 2]
        score[~feasible] = np.inf
        if not np.any(feasible):
            score = 0.58 * f[:, 4] + 0.27 * f[:, 5] + 0.10 * f[:, 3] + 0.05 * f[:, 2]
        return int(np.argmin(score))


class DeepSchedScheduler(BaseScheduler):
    name = "DeepSched"

    def __init__(self, cfg: Dict, node_count: int, seed: int):
        super().__init__(cfg, node_count, seed)
        self.agent = CooperativeLinearQAgent(node_count, int(cfg["marl"]["feature_dimension"]), cfg, seed)
        self.agent.peer_influence = 0.0

    def choose(self, env: CloudEdgeEnvironment, workload: pd.Series) -> int:
        features = env.candidate_features(workload)
        feasible = env.feasible_mask(workload)
        action = self.agent.select_action(features, feasible, training=self.training)
        self._last_features = features
        self._last_action = action
        return action

    def observe(self, env: CloudEdgeEnvironment, workload: pd.Series,
                reward: float, next_workload: Optional[pd.Series]) -> None:
        if not self.training or self._last_features is None:
            return
        next_features = env.candidate_features(next_workload) if next_workload is not None else None
        next_feasible = env.feasible_mask(next_workload) if next_workload is not None else None
        self.agent.update(self._last_action, self._last_features[self._last_action],
                          reward, next_features, next_feasible)


class MARLCoopScheduler(DeepSchedScheduler):
    name = "MARL-Coop"

    def __init__(self, cfg: Dict, node_count: int, seed: int):
        super().__init__(cfg, node_count, seed)
        self.agent.peer_influence = float(cfg["marl"]["peer_influence"])


class LLMHeuristicScheduler(BaseScheduler):
    name = "LLM-Heuristic"

    def __init__(self, cfg: Dict, node_count: int, seed: int, cache_path=None):
        super().__init__(cfg, node_count, seed)
        self.semantic = SemanticAgent(cfg, cache_path)

    def choose(self, env: CloudEdgeEnvironment, workload: pd.Series) -> int:
        weights, latency, hit = self.semantic.infer(workload)
        self.current_weights = weights
        self.last_semantic_latency_ms = latency
        self.last_cache_hit = hit
        costs = env.heuristic_costs(workload)
        score = sum(weights[k] * costs[k] for k in weights)
        feasible = env.feasible_mask(workload)
        score[~feasible] = np.inf
        if not np.any(feasible):
            score = sum(weights[k] * costs[k] for k in weights)
        return int(np.argmin(score))

    def persist(self) -> None:
        self.semantic.persist_cache()


class LLMMARLOrchestrator(MARLCoopScheduler):
    name = "LLM-MARL-Orch"

    def __init__(self, cfg: Dict, node_count: int, seed: int, cache_path=None,
                 use_semantics: bool = True, include_fairness: bool = True,
                 use_drift: bool = True, static_weights: bool = False):
        super().__init__(cfg, node_count, seed)
        self.semantic = SemanticAgent(cfg, cache_path)
        self.use_semantics = use_semantics
        self.include_fairness = include_fairness
        self.use_drift = use_drift
        self.static_weights = static_weights
        self.detector = SemanticDriftDetector(
            cfg["dataset"]["workload_classes"],
            int(cfg["simulation"]["drift_window"]),
            float(cfg["simulation"]["drift_threshold"]),
        )
        self.step_index = 0

    def prepare(self, train_workloads: pd.DataFrame) -> None:
        self.detector.fit_reference(train_workloads["workload_class"].tolist())

    def choose(self, env: CloudEdgeEnvironment, workload: pd.Series) -> int:
        self.step_index += 1
        divergence, triggered = self.detector.update(str(workload["workload_class"]))
        self.last_drift = divergence
        self.last_drift_triggered = bool(triggered and self.use_drift)

        refresh_interval = int(self.cfg["simulation"]["semantic_refresh_interval"])
        force_refresh = self.last_drift_triggered or (self.step_index % refresh_interval == 0)

        if self.use_semantics and not self.static_weights:
            weights, latency, hit = self.semantic.infer(workload, force_refresh=force_refresh)
            self.current_weights = weights
            self.last_semantic_latency_ms = latency
            self.last_cache_hit = hit
        else:
            self.current_weights = dict(DEFAULT_WEIGHTS)
            self.last_semantic_latency_ms = 0.0
            self.last_cache_hit = True

        features = env.candidate_features(workload).copy()
        # Semantic priorities reshape candidate features before Q evaluation.
        features[:, 2] *= 1.0 + self.current_weights["sla"]
        features[:, 4] *= 1.0 + self.current_weights["energy"]
        features[:, 5] *= 1.0 + self.current_weights["carbon"]
        features[:, 8] *= 1.0 + self.current_weights["fairness"]

        feasible = env.feasible_mask(workload)
        action = self.agent.select_action(features, feasible, training=self.training)
        self._last_features = features
        self._last_action = action
        return action

    def persist(self) -> None:
        self.semantic.persist_cache()


def build_scheduler(name: str, cfg: Dict, node_count: int, seed: int, cache_path=None,
                    ablation: str | None = None) -> BaseScheduler:
    if name == "K8s-Default":
        return K8sDefaultScheduler(cfg, node_count, seed)
    if name == "Energy-Aware":
        return EnergyAwareScheduler(cfg, node_count, seed)
    if name == "DeepSched":
        return DeepSchedScheduler(cfg, node_count, seed)
    if name == "MARL-Coop":
        return MARLCoopScheduler(cfg, node_count, seed)
    if name == "LLM-Heuristic":
        return LLMHeuristicScheduler(cfg, node_count, seed, cache_path)
    if name == "LLM-MARL-Orch":
        if ablation == "No-Semantic-Guidance":
            return LLMMARLOrchestrator(cfg, node_count, seed, cache_path, use_semantics=False)
        if ablation == "No-MARL":
            return LLMHeuristicScheduler(cfg, node_count, seed, cache_path)
        if ablation == "No-Fairness":
            return LLMMARLOrchestrator(cfg, node_count, seed, cache_path, include_fairness=False)
        if ablation == "No-Drift-Detection":
            return LLMMARLOrchestrator(cfg, node_count, seed, cache_path, use_drift=False)
        if ablation == "Static-Weights":
            return LLMMARLOrchestrator(cfg, node_count, seed, cache_path, static_weights=True)
        return LLMMARLOrchestrator(cfg, node_count, seed, cache_path)
    raise ValueError(f"Unknown scheduler: {name}")
