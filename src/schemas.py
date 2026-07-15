from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    cluster_id: str
    cluster_kind: str
    node_type: str
    cpu_capacity: float
    memory_capacity_gb: float
    network_capacity_mbps: float
    power_idle_w: float
    power_peak_w: float
    carbon_intensity_g_per_kwh: float
    renewable_ratio: float
    reliability: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    timestamp_s: float
    workload_class: str
    priority: int
    cpu_request: float
    memory_request_gb: float
    network_request_mbps: float
    storage_request_gb: float
    estimated_duration_s: float
    latency_budget_ms: float
    energy_sensitivity: float
    edge_locality: float
    semantic_description: str
    stress_regime: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class StepOutcome:
    node_id: str
    workload_class: str
    response_time_ms: float
    waiting_time_ms: float
    service_time_ms: float
    sla_violation: int
    energy_wh: float
    carbon_g: float
    load_imbalance: float
    fairness_penalty: float
    feasible: int
    reward: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
