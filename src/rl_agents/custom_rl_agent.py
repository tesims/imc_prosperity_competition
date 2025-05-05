# custom_rl_agent.py
import math
import random
import jsonpickle
import numpy as np
import pandas as pd
from typing import List

class RLTradingAgent:
    """
    A linear Q‑learning agent with two actions (0=short, 1=long).
    Q(s,a) = w[a] · features(s)
    """
    def __init__(
        self,
        feature_names: List[str],
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.features = feature_names
        self.n_feats = len(feature_names)
        # one weight vector per action
        self.w = np.zeros((2, self.n_feats), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def featurize(self, row: pd.Series) -> np.ndarray:
        """
        Given a row with columns = self.features, return feature vector.
        """
        return row[self.features].to_numpy(dtype=float)

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        return self.w.dot(phi)

    def choose_action(self, phi: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.choice([0,1])
        return int(np.argmax(self.q_values(phi)))

    def update(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        phi_next: np.ndarray,
        done: bool
    ):
        q_sa = self.q_values(phi)[action]
        q_next = 0.0 if done else np.max(self.q_values(phi_next))
        target = reward + self.gamma * q_next
        error = target - q_sa
        # gradient step
        self.w[action] += self.alpha * error * phi

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(jsonpickle.dumps(self))

    @staticmethod
    def load(path: str) -> "RLTradingAgent":
        with open(path) as f:
            return jsonpickle.loads(f.read())
