import math
import random
import jsonpickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import json

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class EnhancedRLTradingAgent:
    """
    Enhanced Q-learning agent with 5 actions and sophisticated feature extraction.
    Actions: 0=Strong Sell, 1=Weak Sell, 2=Hold, 3=Weak Buy, 4=Strong Buy
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_size: int = 10000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        
        # Initialize replay memory
        self.memory = []
        self.memory_size = memory_size
        
        # Initialize metrics
        self.losses = []
        self.rewards = []
        
        # Trading parameters
        self.position_limit = 50
        self.min_position = 5
        
        # Initialize feature statistics
        self.feature_stats: Dict[str, Tuple[float, float, float]] = {}  # mean, std, max_abs
        self.initialized = False
        
        # Trading history
        self.current_position = 0
        self.trade_history: List[Dict] = []
    
    def initialize_feature_stats(self, data: pd.DataFrame):
        """Initialize feature statistics for normalization."""
        for feature in self.features:
            if feature in data.columns:
                values = data[feature].values
                mean = np.mean(values)
                std = np.std(values)
                max_abs = np.max(np.abs(values))
                self.feature_stats[feature] = (mean, std, max_abs)
        self.initialized = True
    
    def compute_technical_indicators(self, row: pd.Series, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Compute technical indicators for feature enhancement."""
        indicators = []
        
        for feature in self.features:
            if feature not in row:
                continue
                
            value = row[feature]
            
            # Normalize value
            if feature in self.feature_stats:
                mean, std, max_abs = self.feature_stats[feature]
                z_score = (value - mean) / (std + 1e-8)
                norm_value = value / (max_abs + 1e-8)
                indicators.extend([value, z_score, norm_value])
            else:
                indicators.extend([value, 0.0, 0.0])
            
            # Add momentum if history is available
            if history is not None and len(history) > 0:
                momentum = value - history[feature].iloc[-1]
                indicators.append(momentum)
            else:
                indicators.append(0.0)
        
        return np.array(indicators, dtype=float)
    
    def featurize(self, row: pd.Series, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Enhanced feature extraction with technical indicators."""
        # Basic features
        basic_features = row[self.features].to_numpy(dtype=float)
        
        # Technical indicators
        tech_indicators = self.compute_technical_indicators(row, history)
        
        # Position features
        position_features = np.array([
            self.current_position / self.position_limit,
            1.0 if self.current_position > 0 else 0.0,
            1.0 if self.current_position < 0 else 0.0
        ])
        
        # Combine features
        return np.concatenate([basic_features, tech_indicators, position_features])
    
    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        raw_q = self.w.dot(phi)
        
        # Apply position constraints
        if self.current_position >= self.position_limit:
            # Can't buy more
            raw_q[3:] = float('-inf')
        elif self.current_position <= -self.position_limit:
            # Can't sell more
            raw_q[:2] = float('-inf')
        
        return raw_q
    
    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()
    
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> float:
        """Learn from experience tuple."""
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        # Only learn if we have enough samples
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            s, a, r, ns, d = self.memory[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network occasionally
        if len(self.memory) % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store metrics
        self.losses.append(loss.item())
        self.rewards.append(reward)
        
        return loss.item()
    
    def save(self, path: str):
        """Save the agent's state."""
        state = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'metrics': {
                'losses': self.losses,
                'rewards': self.rewards
            }
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load the agent's state."""
        state = torch.load(path)
        
        # Recreate networks if needed
        if not hasattr(self, 'policy_net'):
            self.policy_net = DQN(state['state_size'], state['action_size']).to(self.device)
            self.target_net = DQN(state['state_size'], state['action_size']).to(self.device)
            self.optimizer = optim.Adam(self.policy_net.parameters())
        
        # Load states
        self.policy_net.load_state_dict(state['policy_net_state_dict'])
        self.target_net.load_state_dict(state['target_net_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epsilon = state['epsilon']
        
        # Load metrics
        self.losses = state['metrics']['losses']
        self.rewards = state['metrics']['rewards']
    
    def get_position_size(self, action: int) -> int:
        """Get position size for an action."""
        if action == 0:  # Strong Sell
            return -10
        elif action == 1:  # Weak Sell
            return -5
        elif action == 3:  # Weak Buy
            return 5
        elif action == 4:  # Strong Buy
            return 10
        return 0  # Hold
    
    def update(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        phi_next: np.ndarray,
        done: bool,
        position_delta: int = 0
    ):
        """Update Q-values and agent state."""
        # Update position
        self.current_position += position_delta
        
        # Standard Q-learning update
        q_sa = self.q_values(phi)[action]
        q_next = 0.0 if done else np.max(self.q_values(phi_next))
        target = reward + self.gamma * q_next
        error = target - q_sa
        
        # Gradient step with momentum
        self.w[action] += self.alpha * error * phi
    
    def update_position(self, action: int):
        """Update the current position based on the chosen action."""
        if action == 0:  # Strong Sell
            self.current_position -= 10
        elif action == 1:  # Weak Sell
            self.current_position -= 5
        elif action == 3:  # Weak Buy
            self.current_position += 5
        elif action == 4:  # Strong Buy
            self.current_position += 10
        else:  # Hold
            pass
    
    def reset_position(self):
        """Reset the current position to zero."""
        self.current_position = 0
    
    def add_to_trade_history(self, trade: Dict):
        """Add a trade to the trade history."""
        self.trade_history.append(trade)
    
    def get_trade_history(self) -> List[Dict]:
        """Get the trade history."""
        return self.trade_history

    def get_metrics(self):
        """Get the training metrics."""
        return {
            'losses': self.losses,
            'rewards': self.rewards
        }

    def clone(self):
        """Create a copy of the agent"""
        new_agent = EnhancedRLTradingAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            alpha=0.001,  # Default value, won't affect the copied agent
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            batch_size=self.batch_size,
            memory_size=self.memory_size
        )
        
        # Copy network parameters
        new_agent.policy_net.load_state_dict(self.policy_net.state_dict())
        new_agent.target_net.load_state_dict(self.target_net.state_dict())
        
        # Copy optimizer state
        new_agent.optimizer = optim.Adam(new_agent.policy_net.parameters())
        new_agent.optimizer.load_state_dict(self.optimizer.state_dict())
        
        # Copy other attributes
        new_agent.epsilon = self.epsilon
        new_agent.losses = self.losses.copy() if hasattr(self, 'losses') else []
        new_agent.rewards = self.rewards.copy() if hasattr(self, 'rewards') else []
        
        return new_agent 