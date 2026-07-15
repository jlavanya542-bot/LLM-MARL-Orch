from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from .schemas import StepOutcome


def jain_index(values) -> float:
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0 or np.allclose(x, 0):
        return 1.0
    return float((x.sum() ** 2) / (len(x) * np.sum(x ** 2) + 1e-12))


class MetricTracker:
    def __init__(self):
        self.outcomes: List[StepOutcome] = []
        self.decision_latencies_ms: List[float] = []
        self.semantic_latencies_ms: List[float] = []
        self.cache_hits: List[int] = []
        self.drift_events: List[int] = []
        self.rewards: List[float] = []
        self.class_waits = defaultdict(list)

    def add(self, outcome: StepOutcome, decision_latency_ms: float,
            semantic_latency_ms: float, cache_hit: bool, drift_triggered: bool) -> None:
        self.outcomes.append(outcome)
        self.decision_latencies_ms.append(float(decision_latency_ms))
        self.semantic_latencies_ms.append(float(semantic_latency_ms))
        self.cache_hits.append(int(cache_hit))
        self.drift_events.append(int(drift_triggered))
        self.rewards.append(float(outcome.reward))
        self.class_waits[outcome.workload_class].append(outcome.waiting_time_ms)

    def summary(self) -> Dict[str, float]:
        if not self.outcomes:
            raise RuntimeError("No outcomes recorded")
        o = self.outcomes
        class_service = []
        for waits in self.class_waits.values():
            mean_wait = float(np.mean(waits)) if waits else 0.0
            class_service.append(1.0 / (1.0 + mean_wait))

        return {
            "sla_violation_rate_pct": 100.0 * float(np.mean([x.sla_violation for x in o])),
            "energy_wh_per_container": float(np.mean([x.energy_wh for x in o])),
            "carbon_g_per_container": float(np.mean([x.carbon_g for x in o])),
            "mean_response_time_ms": float(np.mean([x.response_time_ms for x in o])),
            "p95_response_time_ms": float(np.percentile([x.response_time_ms for x in o], 95)),
            "scheduling_latency_ms": float(np.mean(self.decision_latencies_ms)),
            "p95_scheduling_latency_ms": float(np.percentile(self.decision_latencies_ms, 95)),
            "semantic_refresh_latency_ms": float(np.mean([x for x in self.semantic_latencies_ms if x > 0]))
                if any(x > 0 for x in self.semantic_latencies_ms) else 0.0,
            "semantic_cache_hit_rate": float(np.mean(self.cache_hits)),
            "fairness_index": jain_index(class_service),
            "load_imbalance_score": float(np.mean([x.load_imbalance for x in o])),
            "feasible_placement_rate_pct": 100.0 * float(np.mean([x.feasible for x in o])),
            "mean_reward": float(np.mean(self.rewards)),
            "drift_event_count": int(np.sum(self.drift_events)),
            "evaluated_containers": int(len(o)),
        }
