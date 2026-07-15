from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .cloud_edge_environment import CloudEdgeEnvironment
from .metrics import MetricTracker
from .reward_model import MultiObjectiveReward
from .schedulers import build_scheduler
from .trainer import train_scheduler
from .workload_generator import generate_nodes, generate_workloads


def temporal_split(workloads: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(workloads)
    train_end = int(n * float(cfg["dataset"]["train_fraction"]))
    val_end = train_end + int(n * float(cfg["dataset"]["validation_fraction"]))
    return (
        workloads.iloc[:train_end].reset_index(drop=True),
        workloads.iloc[train_end:val_end].reset_index(drop=True),
        workloads.iloc[val_end:].reset_index(drop=True),
    )


def evaluate_scheduler(scheduler_name: str, cfg: Dict, seed: int,
                       workload_count: int, cache_path: Path,
                       ablation: str | None = None) -> Dict[str, float]:
    nodes = generate_nodes(cfg, seed)
    workloads = generate_workloads(cfg, seed, workload_count)
    train, validation, test = temporal_split(workloads, cfg)

    scheduler = build_scheduler(
        scheduler_name, cfg, len(nodes), seed,
        cache_path=cache_path, ablation=ablation
    )
    train_info = train_scheduler(scheduler, nodes, train, cfg)

    env = CloudEdgeEnvironment(nodes, cfg)
    reward_model = MultiObjectiveReward(cfg)
    tracker = MetricTracker()
    scheduler.set_training(False)
    scheduler.prepare(train)

    for i in range(len(test)):
        row = test.iloc[i]
        next_row = test.iloc[i + 1] if i + 1 < len(test) else None
        start = time.perf_counter()
        action = scheduler.choose(env, row)
        decision_latency_ms = (time.perf_counter() - start) * 1000.0

        outcome = env.step(action, row)
        include_fairness = getattr(scheduler, "include_fairness", True)
        reward, _ = reward_model.compute(outcome, scheduler.current_weights, include_fairness)
        outcome.reward = reward
        scheduler.observe(env, row, reward, next_row)

        tracker.add(
            outcome,
            decision_latency_ms=decision_latency_ms,
            semantic_latency_ms=scheduler.last_semantic_latency_ms,
            cache_hit=scheduler.last_cache_hit,
            drift_triggered=scheduler.last_drift_triggered,
        )

    scheduler.persist()
    result = tracker.summary()
    result.update(train_info)
    result.update({
        "scheduler": scheduler_name if ablation is None else f"LLM-MARL-Orch::{ablation}",
        "seed": int(seed),
        "train_count": int(len(train)),
        "validation_count": int(len(validation)),
        "test_count": int(len(test)),
    })
    return result


def run_benchmark(cfg: Dict, seeds: Iterable[int], workload_count: int,
                  output_dir: str | Path, include_ablations: bool = True) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "semantic_weight_cache.json"
    rows: List[Dict[str, float]] = []

    for seed in seeds:
        for scheduler_name in cfg["experiment"]["schedulers"]:
            rows.append(evaluate_scheduler(
                scheduler_name, cfg, int(seed), int(workload_count), cache_path
            ))

        if include_ablations:
            for ablation in cfg["experiment"]["ablations"]:
                if ablation == "Full":
                    continue
                rows.append(evaluate_scheduler(
                    "LLM-MARL-Orch", cfg, int(seed), int(workload_count),
                    cache_path, ablation=ablation
                ))

    return pd.DataFrame(rows)
