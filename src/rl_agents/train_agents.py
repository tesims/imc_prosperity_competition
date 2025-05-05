# train_agents.py
import os
import pandas as pd
from custom_rl_agent import RLTradingAgent
from simple_env       import TradingEnv

PRODUCTS = {

  
}

FEATURES = ["prediction","r","price"]  # your chosen features; must exist in CSV

def train_one(product: str, path: str, episodes: int = 200):
    df = pd.read_csv(path)
    # assume your CSV already has a 'prediction' column
    env = TradingEnv(df, lags=1)
    agent = RLTradingAgent(FEATURES, alpha=0.01, gamma=0.99, epsilon=0.1)

    for ep in range(episodes):
        state_row, _ = env.reset()
        phi = agent.featurize(state_row)
        total_reward = 0.0

        while True:
            action = agent.choose_action(phi)
            next_row, reward, done, _ = env.step(action)
            phi_next = agent.featurize(next_row)
            agent.update(phi, action, reward, phi_next, done)
            phi = phi_next
            total_reward += reward
            if done:
                print(f"{product} | ep {ep} | reward {total_reward:.3f}")
                break

    # save per‑product agent
    os.makedirs("models", exist_ok=True)
    agent.save(f"models/{product}_agent.jsonpickle")

def main():
    for prod, path in PRODUCTS.items():
        train_one(prod, path, episodes=300)

if __name__=="__main__":
    main()
