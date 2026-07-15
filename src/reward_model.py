from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .schemas import StepOutcome


class MultiObjectiveReward:
    """Scale-compatible reward used by every learning scheduler."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def compute(self, outcome: StepOutcome, weights: Dict[str, float],
                include_fairness: bool = True) -> Tuple[float, Dict[str, float]]:
        sim = self.cfg["simulation"]
        costs = {
            "sla": float(outcome.sla_violation),
            "energy": float(np.clip(outcome.energy_wh / float(sim["energy_reference_wh"]), 0.0, 3.0)),
            "imbalance": float(np.clip(outcome.load_imbalance, 0.0, 3.0)),
            "fairness": float(np.clip(outcome.fairness_penalty, 0.0, 3.0)) if include_fairness else 0.0,
            "carbon": float(np.clip(outcome.carbon_g / float(sim["carbon_reference_g"]), 0.0, 3.0)),
        }
        weighted_cost = sum(float(weights[k]) * costs[k] for k in costs)
        feasibility_bonus = 0.20 if outcome.feasible else -1.50
        reward = feasibility_bonus - weighted_cost
        components = {f"cost_{k}": v for k, v in costs.items()}
        components["weighted_cost"] = float(weighted_cost)
        components["feasibility_bonus"] = float(feasibility_bonus)
        return float(reward), components
