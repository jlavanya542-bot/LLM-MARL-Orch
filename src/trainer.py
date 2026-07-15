from __future__ import annotations

from typing import Dict

import pandas as pd

from .cloud_edge_environment import CloudEdgeEnvironment
from .reward_model import MultiObjectiveReward
from .schedulers import BaseScheduler


def train_scheduler(scheduler: BaseScheduler, nodes: pd.DataFrame,
                    workloads: pd.DataFrame, cfg: Dict) -> Dict[str, float]:
    """Train adaptive schedulers on the chronological training partition."""
    scheduler.set_training(True)
    scheduler.prepare(workloads)
    env = CloudEdgeEnvironment(nodes, cfg)
    reward_model = MultiObjectiveReward(cfg)
    total_reward = 0.0

    for i in range(len(workloads)):
        row = workloads.iloc[i]
        next_row = workloads.iloc[i + 1] if i + 1 < len(workloads) else None
        action = scheduler.choose(env, row)
        outcome = env.step(action, row)
        include_fairness = getattr(scheduler, "include_fairness", True)
        reward, _ = reward_model.compute(outcome, scheduler.current_weights, include_fairness)
        outcome.reward = reward
        scheduler.observe(env, row, reward, next_row)
        total_reward += reward

    scheduler.set_training(False)
    return {
        "training_steps": int(len(workloads)),
        "mean_training_reward": float(total_reward / max(len(workloads), 1)),
    }
