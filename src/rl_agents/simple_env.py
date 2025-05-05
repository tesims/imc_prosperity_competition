# simple_env.py
import pandas as pd
import numpy as np
import random
from typing import Tuple

class TradingEnv:
    """
    Takes a DataFrame with columns:
      - 'price'        : actual price
      - 'prediction'   : a historic predicted return or direction
      plus any other features you like.
    At each step, action ∈ {0 (short), 1 (long)}.
    Reward = realized log‐return if correct direction, else negative.
    """
    def __init__(self, df: pd.DataFrame, lags: int = 1):
        self.df = df.reset_index(drop=True)
        self.lags = lags
        self.n = len(df)
        # precompute true direction: 1 if next log‑return >0 else 0
        self.df["r"] = np.log(self.df["price"] / self.df["price"].shift(1))
        self.df.dropna(inplace=True)
        self.df["d"] = (self.df["r"] > 0).astype(int).values
        self.reset()

    def reset(self) -> Tuple[pd.Series,int]:
        self.t = self.lags
        self.done = False
        return self._get_state(), {}

    def _get_state(self) -> pd.Series:
        # return the row at time t for feature extraction
        return self.df.iloc[self.t]

    def step(self, action: int) -> Tuple[pd.Series,float,bool,dict]:
        true_dir = int(self.df.at[self.t, "d"])
        reward = abs(self.df.at[self.t, "r"]) if action == true_dir else -abs(self.df.at[self.t, "r"])
        self.t += 1
        if self.t >= len(self.df):
            self.done = True
        return (self._get_state(), reward, self.done, {})
