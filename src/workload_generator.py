from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


_CLUSTER_LAYOUT = [
    ("cloud-a", "cloud", "CO", 25),
    ("cloud-b", "cloud", "BA", 20),
    ("edge-a", "edge", "EE", 30),
    ("edge-b", "edge", "BA", 25),
]

_NODE_BASE = {
    "CO": dict(cpu=32, memory=128, network=10000, idle=105, peak=310, carbon=430, renewable=0.18, reliability=0.997),
    "BA": dict(cpu=16, memory=64, network=5000, idle=70, peak=185, carbon=360, renewable=0.30, reliability=0.994),
    "EE": dict(cpu=8, memory=32, network=2500, idle=25, peak=82, carbon=185, renewable=0.64, reliability=0.989),
}

_WORKLOAD_PARAMS = {
    "LS": dict(cpu=(0.5, 3.0), memory=(0.5, 5.0), network=(80, 600), storage=(0.1, 2.0),
               duration=(0.08, 0.45), latency=(35, 60), energy=(0.35, 0.65), locality=(0.75, 1.0)),
    "TC": dict(cpu=(2.0, 8.0), memory=(4.0, 20.0), network=(100, 900), storage=(2.0, 20.0),
               duration=(0.8, 4.0), latency=(220, 340), energy=(0.70, 1.0), locality=(0.0, 0.35)),
    "TX": dict(cpu=(0.6, 3.5), memory=(1.0, 7.0), network=(60, 450), storage=(0.2, 3.5),
               duration=(0.12, 0.70), latency=(75, 125), energy=(0.20, 0.50), locality=(0.25, 0.70)),
    "HB": dict(cpu=(1.0, 6.0), memory=(2.0, 14.0), network=(70, 750), storage=(0.5, 10.0),
               duration=(0.25, 2.5), latency=(80, 260), energy=(0.35, 0.90), locality=(0.20, 0.85)),
}


def generate_nodes(cfg: Dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    idx = 0
    for cluster_id, cluster_kind, node_type, count in _CLUSTER_LAYOUT:
        base = _NODE_BASE[node_type]
        for _ in range(count):
            jitter = rng.normal(1.0, 0.06)
            rows.append({
                "node_id": f"node-{idx:03d}",
                "cluster_id": cluster_id,
                "cluster_kind": cluster_kind,
                "node_type": node_type,
                "cpu_capacity": round(base["cpu"] * max(jitter, 0.82), 3),
                "memory_capacity_gb": round(base["memory"] * max(rng.normal(1.0, 0.05), 0.85), 3),
                "network_capacity_mbps": round(base["network"] * max(rng.normal(1.0, 0.08), 0.75), 3),
                "power_idle_w": round(base["idle"] * max(rng.normal(1.0, 0.04), 0.85), 3),
                "power_peak_w": round(base["peak"] * max(rng.normal(1.0, 0.05), 0.85), 3),
                "carbon_intensity_g_per_kwh": round(base["carbon"] * max(rng.normal(1.0, 0.10), 0.55), 3),
                "renewable_ratio": round(float(np.clip(rng.normal(base["renewable"], 0.08), 0.0, 0.95)), 4),
                "reliability": round(float(np.clip(rng.normal(base["reliability"], 0.002), 0.96, 0.9999)), 5),
            })
            idx += 1

    nodes = pd.DataFrame(rows)
    expected = int(cfg["dataset"]["node_count"])
    if expected != len(nodes):
        if expected < len(nodes):
            nodes = nodes.iloc[:expected].copy()
        else:
            extra = nodes.sample(expected - len(nodes), replace=True, random_state=seed).copy()
            extra["node_id"] = [f"node-{i:03d}" for i in range(len(nodes), expected)]
            nodes = pd.concat([nodes, extra], ignore_index=True)
    return nodes


def _regime(index: int, total: int) -> str:
    ratio = index / max(total - 1, 1)
    if ratio < 0.25:
        return "normal"
    if ratio < 0.50:
        return "high-contention"
    if ratio < 0.75:
        return "burst"
    return "energy-scarcity"


def generate_workloads(cfg: Dict, seed: int, count: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    n = int(count or cfg["dataset"]["workload_count"])
    classes = np.array(cfg["dataset"]["workload_classes"])
    probabilities = np.array([0.28, 0.24, 0.28, 0.20])
    chosen = rng.choice(classes, size=n, p=probabilities)

    interarrival = rng.exponential(scale=17.28, size=n)
    timestamps = np.cumsum(interarrival)
    timestamps *= (24 * 3600) / max(timestamps[-1], 1.0)

    rows = []
    for i, cls in enumerate(chosen):
        p = _WORKLOAD_PARAMS[str(cls)]
        regime = _regime(i, n)
        stress_multiplier = {"normal": 1.0, "high-contention": 1.12, "burst": 1.35, "energy-scarcity": 1.08}[regime]
        if regime == "burst" and rng.random() < 0.28:
            timestamps[i] = max(0.0, timestamps[i] - rng.uniform(0, 20))

        priority = int(rng.choice([1, 2, 3], p=[0.30, 0.48, 0.22]))
        cpu = rng.uniform(*p["cpu"]) * stress_multiplier
        mem = rng.uniform(*p["memory"]) * stress_multiplier
        net = rng.uniform(*p["network"]) * stress_multiplier
        storage = rng.uniform(*p["storage"])
        duration = rng.uniform(*p["duration"]) * stress_multiplier
        latency = rng.uniform(*p["latency"])
        energy = rng.uniform(*p["energy"])
        locality = rng.uniform(*p["locality"])

        descriptors = {
            "LS": "latency-sensitive stream requiring edge locality and rapid response",
            "TC": "throughput-centric batch workload with high compute and energy sensitivity",
            "TX": "transactional service requiring stable latency and low jitter",
            "HB": "hybrid microservice chain with variable compute and latency demand",
        }
        semantic = (
            f"{descriptors[str(cls)]}; priority={priority}; regime={regime}; "
            f"cpu={cpu:.2f}; memory_gb={mem:.2f}; latency_budget_ms={latency:.1f}; "
            f"energy_sensitivity={energy:.2f}; edge_locality={locality:.2f}"
        )
        rows.append({
            "workload_id": f"workload-{i:06d}",
            "timestamp_s": round(float(timestamps[i]), 5),
            "workload_class": str(cls),
            "priority": priority,
            "cpu_request": round(float(cpu), 4),
            "memory_request_gb": round(float(mem), 4),
            "network_request_mbps": round(float(net), 4),
            "storage_request_gb": round(float(storage), 4),
            "estimated_duration_s": round(float(duration), 5),
            "latency_budget_ms": round(float(latency), 4),
            "energy_sensitivity": round(float(energy), 4),
            "edge_locality": round(float(locality), 4),
            "semantic_description": semantic,
            "stress_regime": regime,
        })

    return pd.DataFrame(rows).sort_values("timestamp_s").reset_index(drop=True)


def semantic_target_weights(workload_row: pd.Series, cfg: Dict) -> Dict[str, float]:
    base = dict(cfg["semantic_agent"]["weights"][str(workload_row["workload_class"])])
    priority = int(workload_row["priority"])
    if priority == 3:
        base["sla"] += 0.06
        base["energy"] -= 0.03
        base["carbon"] -= 0.03
    if str(workload_row["stress_regime"]) == "energy-scarcity":
        base["energy"] += 0.05
        base["carbon"] += 0.03
        base["sla"] -= 0.05
        base["imbalance"] -= 0.03
    if float(workload_row["edge_locality"]) > 0.80:
        base["sla"] += 0.03
        base["imbalance"] -= 0.01
        base["fairness"] -= 0.02
    values = np.clip(np.array(list(base.values()), dtype=float), 0.01, None)
    values /= values.sum()
    return dict(zip(base.keys(), values.round(6)))


def generate_prompt_corpus(cfg: Dict, workloads: pd.DataFrame, seed: int,
                           count: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2000)
    n = int(count or cfg["dataset"]["prompt_corpus_size"])
    indices = rng.integers(0, len(workloads), size=n)
    rows = []
    for prompt_id, idx in enumerate(indices):
        row = workloads.iloc[int(idx)]
        target = semantic_target_weights(row, cfg)
        rows.append({
            "prompt_id": f"prompt-{prompt_id:06d}",
            "source_workload_id": row["workload_id"],
            "prompt": (
                "Interpret the scheduling intent and return normalized priorities for "
                "SLA, energy, imbalance, fairness, and carbon. Context: "
                + row["semantic_description"]
            ),
            **{f"target_{k}": v for k, v in target.items()},
        })
    return pd.DataFrame(rows)


def save_public_dataset(cfg: Dict, output_dir: str | Path, seed: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(seed if seed is not None else cfg["reproducibility"]["master_seed"])

    nodes = generate_nodes(cfg, seed)
    workloads = generate_workloads(cfg, seed)
    prompts = generate_prompt_corpus(cfg, workloads, seed)

    nodes.to_csv(output / "nodes.csv", index=False)
    workloads.to_csv(output / "workloads.csv", index=False)
    prompts.to_csv(output / "semantic_prompt_corpus.csv", index=False)

    metadata = {
        "dataset_name": "LLM-MARL-Orch Synthetic Cloud-Edge Scheduling Dataset",
        "version": "1.0.0",
        "generator_seed": seed,
        "node_count": int(len(nodes)),
        "workload_count": int(len(workloads)),
        "prompt_count": int(len(prompts)),
        "contains_personal_data": False,
        "temporal_split": {
            "train_fraction": cfg["dataset"]["train_fraction"],
            "validation_fraction": cfg["dataset"]["validation_fraction"],
            "test_fraction": cfg["dataset"]["test_fraction"],
            "overlap": False,
        },
        "source_inspiration": [
            "Public cluster-trace workload patterns",
            "Cloud-native microservice scheduling constraints",
            "Synthetic cloud-edge sustainability scenarios",
        ],
    }
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    schema = {
        "nodes.csv": {c: str(nodes[c].dtype) for c in nodes.columns},
        "workloads.csv": {c: str(workloads[c].dtype) for c in workloads.columns},
        "semantic_prompt_corpus.csv": {c: str(prompts[c].dtype) for c in prompts.columns},
    }
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return nodes, workloads, prompts
