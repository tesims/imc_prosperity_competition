import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from enhanced_rl_agent import EnhancedRLTradingAgent
from enhanced_env import EnhancedTradingEnv

def test_agent(
    product: str,
    data: pd.DataFrame,
    features: list,
    episodes: int = 100
):
    """Test an enhanced RL agent on a product."""
    print(f"\nTesting agent for {product}")
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    # Initialize agent
    agent = EnhancedRLTradingAgent(
        feature_names=features,
        alpha=0.01,
        gamma=0.99,
        epsilon=0.1
    )
    agent.initialize_feature_stats(train_data)
    
    # Training environment
    env = EnhancedTradingEnv(
        df=train_data,
        position_limit=50,
        transaction_cost=0.001
    )
    
    # Training loop
    print("\nTraining agent...")
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            history = env.get_history()
            phi = agent.featurize(state, history)
            action = agent.choose_action(phi)
            
            next_state, reward, done, info = env.step(action)
            next_history = env.get_history()
            phi_next = agent.featurize(next_state, next_history)
            
            position_delta = info['position']
            agent.update(phi, action, reward, phi_next, done, position_delta)
            
            total_reward += reward
            if done:
                break
            state = next_state
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | Reward: {total_reward:.2f}")
    
    # Testing
    print("\nTesting agent...")
    test_env = EnhancedTradingEnv(
        df=test_data,
        position_limit=50,
        transaction_cost=0.001
    )
    
    # Test metrics
    rewards = []
    positions = []
    actions = []
    
    state, _ = test_env.reset()
    while True:
        history = test_env.get_history()
        phi = agent.featurize(state, history)
        action = agent.choose_action(phi)
        
        next_state, reward, done, info = test_env.step(action)
        
        rewards.append(reward)
        positions.append(info['position'])
        actions.append(action)
        
        if done:
            break
        state = next_state
    
    # Print results
    print("\nTest Results:")
    print(f"Total Reward: {sum(rewards):.2f}")
    print(f"Sharpe Ratio: {np.mean(rewards)/(np.std(rewards) + 1e-6) * np.sqrt(252):.2f}")
    print(f"Max Position: {max(map(abs, positions))}")
    print(f"Number of Trades: {sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(positions)
    plt.title('Positions')
    plt.xlabel('Step')
    plt.ylabel('Position')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(rewards))
    plt.title('Cumulative Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.show()
    
    return agent

def main():
    # Configuration
    PRODUCTS = ["KELP", "SQUID_INK", "RAINFOREST_RESIN"]
    FEATURES = ["price", "volume", "prediction", "r"]
    DATA_DIR = "data/features/round_four"
    MODELS_DIR = "models/enhanced"
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for product in PRODUCTS:
        # Load data
        data_path = Path(DATA_DIR) / f"{product}_features.csv"
        if not data_path.exists():
            print(f"No data found for {product}")
            continue
        
        data = pd.read_csv(data_path)
        
        # Test agent
        agent = test_agent(
            product=product,
            data=data,
            features=FEATURES,
            episodes=100
        )
        
        # Save agent
        agent.save(str(Path(MODELS_DIR) / f"{product}_agent.jsonpickle"))

if __name__ == "__main__":
    main() 