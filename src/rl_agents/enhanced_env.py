import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List, Any
from collections import deque
import gym
from gym import spaces

class EnhancedTradingEnv(gym.Env):
    """
    Enhanced trading environment with:
    - Position tracking
    - Transaction costs
    - More sophisticated reward function
    - Historical state management
    """
    def __init__(
        self,
        df: pd.DataFrame,
        position_limit: int = 100,
        transaction_cost: float = 0.001,
        history_window: int = 10
    ):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with price and feature data
            position_limit: Maximum absolute position size
            transaction_cost: Cost per trade as fraction of price
            history_window: Number of past states to keep
        """
        super().__init__()
        
        self.df = df
        self.position_limit = position_limit
        self.transaction_cost = transaction_cost
        self.history_window = history_window
        
        # Define action space: -2 (strong sell), -1 (sell), 0 (hold), 1 (buy), 2 (strong buy)
        self.action_space = spaces.Discrete(5)
        
        # Define observation space
        n_features = len(df.columns) + 2  # Add position and PnL
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset the environment."""
        self.current_step = self.history_window
        self.position = 0
        self.cash = 0
        self.pnl = 0
        self.trades = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        features = self.df.iloc[self.current_step].values
        
        # Add position and PnL to observation
        obs = np.append(features, [self.position / self.position_limit, self.pnl])
        return obs.astype(np.float32)
    
    def _calculate_reward(self, old_position: int, new_position: int) -> float:
        """Calculate reward based on position change and price movement."""
        # Get price changes
        current_price = self.df.iloc[self.current_step]["mid_price"]
        next_price = self.df.iloc[self.current_step + 1]["mid_price"]
        price_change = next_price - current_price
        
        # Calculate position reward
        position_reward = new_position * price_change
        
        # Calculate transaction cost
        position_change = abs(new_position - old_position)
        transaction_cost = position_change * current_price * self.transaction_cost
        
        return position_reward - transaction_cost
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        assert self.action_space.contains(action)
        
        # Map action to position change
        action_to_position = {
            0: -2,  # Strong sell
            1: -1,  # Sell
            2: 0,   # Hold
            3: 1,   # Buy
            4: 2    # Strong buy
        }
        
        # Calculate new position
        position_change = action_to_position[action] * (self.position_limit // 2)
        old_position = self.position
        self.position = np.clip(
            self.position + position_change,
            -self.position_limit,
            self.position_limit
        )
        
        # Calculate reward
        reward = self._calculate_reward(old_position, self.position)
        self.pnl += reward
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Store trade information
        if old_position != self.position:
            self.trades.append({
                'timestamp': self.df.index[self.current_step],
                'action': action,
                'position_change': self.position - old_position,
                'price': self.df.iloc[self.current_step]["mid_price"],
                'pnl': self.pnl
            })
        
        info = {
            'position': self.position,
            'pnl': self.pnl,
            'trades': self.trades
        }
        
        return obs, reward, done, info
    
    def render(self) -> None:
        """Print current state information."""
        info = self._get_info()
        print(f"\nStep: {info['step']}")
        print(f"Position: {info['position']}")
        print(f"Price: {info['price']:.2f}")
        print(f"Returns: {info['returns']:.4f}")
        print(f"Volatility: {info['volatility']:.4f}")
    
    def get_history(self) -> pd.DataFrame:
        """Get historical states as DataFrame."""
        return pd.DataFrame(list(self.history)) 