from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


class CooperativeLinearQAgent:
    """Decentralized node agents with a shared global reward signal."""

    def __init__(self, node_count: int, feature_dim: int, cfg: Dict, seed: int):
        marl = cfg["marl"]
        self.node_count = int(node_count)
        self.feature_dim = int(feature_dim)
        self.lr = float(marl["learning_rate"])
        self.gamma = float(marl["discount_factor"])
        self.epsilon = float(marl["epsilon_start"])
        self.epsilon_end = float(marl["epsilon_end"])
        self.epsilon_decay = float(marl["epsilon_decay"])
        self.peer_influence = float(marl["peer_influence"])
        self.gradient_clip = float(marl["gradient_clip"])
        self.l2 = float(marl["l2_regularization"])
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.normal(0.0, 0.025, size=(self.node_count, self.feature_dim))
        self.bias = np.zeros(self.node_count, dtype=float)
        self.last_q = np.zeros(self.node_count, dtype=float)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        local_q = np.sum(self.theta * features, axis=1) + self.bias
        peer_signal = np.mean(local_q) - local_q
        q = local_q + self.peer_influence * peer_signal
        self.last_q = q
        return q

    def select_action(self, features: np.ndarray, feasible_mask: np.ndarray,
                      training: bool = True) -> int:
        feasible = np.flatnonzero(feasible_mask)
        if len(feasible) == 0:
            return int(np.argmin(features[:, 0] + features[:, 1]))

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(feasible))

        q = self.q_values(features).copy()
        q[~feasible_mask] = -np.inf
        return int(np.argmax(q))

    def update(self, action: int, feature: np.ndarray, reward: float,
               next_features: np.ndarray | None = None,
               next_feasible: np.ndarray | None = None) -> float:
        q_current = float(np.dot(self.theta[action], feature) + self.bias[action])
        if next_features is not None and next_feasible is not None and np.any(next_feasible):
            next_q = self.q_values(next_features).copy()
            next_q[~next_feasible] = -np.inf
            bootstrap = float(np.max(next_q))
        else:
            bootstrap = 0.0

        target = float(reward) + self.gamma * bootstrap
        td_error = float(np.clip(target - q_current, -self.gradient_clip, self.gradient_clip))
        grad = td_error * feature - self.l2 * self.theta[action]
        self.theta[action] += self.lr * grad
        self.bias[action] += self.lr * td_error
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return td_error

    def state_dict(self) -> Dict[str, object]:
        return {
            "theta": self.theta.tolist(),
            "bias": self.bias.tolist(),
            "epsilon": float(self.epsilon),
        }
