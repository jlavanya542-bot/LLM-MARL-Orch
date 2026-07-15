from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .schemas import StepOutcome


class CloudEdgeEnvironment:
    """Deterministic event-based simulator for hybrid cloud-edge placement."""

    def __init__(self, nodes: pd.DataFrame, cfg: Dict):
        self.nodes = nodes.reset_index(drop=True).copy()
        self.cfg = cfg
        self.node_count = len(self.nodes)
        self.decay = float(cfg["simulation"]["node_decay"])
        self.reset()

    def reset(self) -> None:
        self.queue_end_s = np.zeros(self.node_count, dtype=float)
        self.dynamic_load = np.zeros(self.node_count, dtype=float)
        self.assignment_count = np.zeros(self.node_count, dtype=float)
        self.class_waits = defaultdict(list)
        self.total_steps = 0

    def feasible_mask(self, workload: pd.Series) -> np.ndarray:
        cpu_ok = self.nodes["cpu_capacity"].to_numpy(float) >= float(workload["cpu_request"])
        mem_ok = self.nodes["memory_capacity_gb"].to_numpy(float) >= float(workload["memory_request_gb"])
        net_ok = self.nodes["network_capacity_mbps"].to_numpy(float) >= float(workload["network_request_mbps"])
        return cpu_ok & mem_ok & net_ok

    def candidate_features(self, workload: pd.Series) -> np.ndarray:
        timestamp = float(workload["timestamp_s"])
        cpu_fraction = float(workload["cpu_request"]) / self.nodes["cpu_capacity"].to_numpy(float)
        mem_fraction = float(workload["memory_request_gb"]) / self.nodes["memory_capacity_gb"].to_numpy(float)
        queue_delay_s = np.maximum(self.queue_end_s - timestamp, 0.0)
        queue_norm = np.clip(queue_delay_s / 2.0, 0.0, 3.0)
        projected_load = np.clip(self.dynamic_load * self.decay + 0.55 * cpu_fraction + 0.25 * mem_fraction, 0.0, 2.5)

        power_range = self.nodes["power_peak_w"].to_numpy(float) - self.nodes["power_idle_w"].to_numpy(float)
        incremental_power = self.nodes["power_idle_w"].to_numpy(float) + power_range * np.clip(projected_load, 0.0, 1.0)
        energy_wh = incremental_power * float(workload["estimated_duration_s"]) / 3600.0
        carbon_g = (
            energy_wh / 1000.0
            * self.nodes["carbon_intensity_g_per_kwh"].to_numpy(float)
            * (1.0 - self.nodes["renewable_ratio"].to_numpy(float))
        )
        edge = (self.nodes["cluster_kind"].to_numpy(str) == "edge").astype(float)
        locality_penalty = np.abs(edge - float(workload["edge_locality"]))
        reliability_penalty = 1.0 - self.nodes["reliability"].to_numpy(float)
        assignment_norm = self.assignment_count / max(self.assignment_count.max(), 1.0)

        features = np.column_stack([
            np.clip(cpu_fraction, 0, 2),
            np.clip(mem_fraction, 0, 2),
            queue_norm,
            projected_load,
            np.clip(energy_wh / float(self.cfg["simulation"]["energy_reference_wh"]), 0, 3),
            np.clip(carbon_g / float(self.cfg["simulation"]["carbon_reference_g"]), 0, 3),
            locality_penalty,
            reliability_penalty * 100.0,
            assignment_norm,
        ])
        return features

    def heuristic_costs(self, workload: pd.Series) -> Dict[str, np.ndarray]:
        f = self.candidate_features(workload)
        return {
            "sla": np.clip(0.45 * f[:, 2] + 0.35 * f[:, 3] + 0.20 * f[:, 6], 0, 3),
            "energy": f[:, 4],
            "imbalance": np.abs(f[:, 3] - np.mean(f[:, 3])),
            "fairness": f[:, 8],
            "carbon": f[:, 5],
        }

    def step(self, node_index: int, workload: pd.Series) -> StepOutcome:
        feasible = bool(self.feasible_mask(workload)[node_index])
        timestamp = float(workload["timestamp_s"])
        node = self.nodes.iloc[int(node_index)]

        waiting_s = max(float(self.queue_end_s[node_index]) - timestamp, 0.0)
        cpu_fraction = float(workload["cpu_request"]) / float(node["cpu_capacity"])
        mem_fraction = float(workload["memory_request_gb"]) / float(node["memory_capacity_gb"])
        net_fraction = float(workload["network_request_mbps"]) / float(node["network_capacity_mbps"])

        load_before = float(self.dynamic_load[node_index])
        service_scale = 0.055 + 0.13 * cpu_fraction + 0.07 * mem_fraction + 0.04 * net_fraction + 0.08 * load_before
        service_time_ms = max(2.0, float(workload["estimated_duration_s"]) * 1000.0 * service_scale)
        if str(node["cluster_kind"]) == "cloud" and float(workload["edge_locality"]) > 0.70:
            service_time_ms += 18.0 + 25.0 * float(workload["edge_locality"])
        if not feasible:
            service_time_ms += 1000.0

        response_time_ms = waiting_s * 1000.0 + service_time_ms
        sla_violation = int(response_time_ms > float(workload["latency_budget_ms"]))

        projected_load = np.clip(load_before * self.decay + 0.60 * cpu_fraction + 0.25 * mem_fraction, 0.0, 2.5)
        utilization = min(projected_load, 1.0)
        power_w = float(node["power_idle_w"]) + (
            float(node["power_peak_w"]) - float(node["power_idle_w"])
        ) * utilization
        energy_wh = power_w * (service_time_ms / 1000.0) / 3600.0
        carbon_g = (
            energy_wh / 1000.0
            * float(node["carbon_intensity_g_per_kwh"])
            * (1.0 - float(node["renewable_ratio"]))
        )

        self.dynamic_load *= self.decay
        self.dynamic_load[node_index] = projected_load
        self.queue_end_s[node_index] = max(self.queue_end_s[node_index], timestamp) + service_time_ms / 1000.0
        self.assignment_count[node_index] += 1.0
        self.total_steps += 1

        normalized = self.dynamic_load / max(np.mean(self.dynamic_load) + 1e-8, 1e-8)
        load_imbalance = float(np.std(normalized))

        cls = str(workload["workload_class"])
        wait_ms = waiting_s * 1000.0
        self.class_waits[cls].append(wait_ms)
        class_means = [np.mean(v) for v in self.class_waits.values() if v]
        fairness_penalty = float(np.std(class_means) / (np.mean(class_means) + 1.0)) if len(class_means) > 1 else 0.0

        return StepOutcome(
            node_id=str(node["node_id"]),
            workload_class=cls,
            response_time_ms=float(response_time_ms),
            waiting_time_ms=float(wait_ms),
            service_time_ms=float(service_time_ms),
            sla_violation=sla_violation,
            energy_wh=float(energy_wh),
            carbon_g=float(carbon_g),
            load_imbalance=load_imbalance,
            fairness_penalty=fairness_penalty,
            feasible=int(feasible),
        )
