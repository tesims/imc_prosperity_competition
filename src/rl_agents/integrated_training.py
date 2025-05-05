import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from synthetic_data.gan import TimeSeriesGAN
from synthetic_data.train import TimeSeriesTrainer
from custom_rl_agent import RLTradingAgent
from simple_env import TradingEnv

class IntegratedTrainingPipeline:
    def __init__(
        self,
        products: List[str],
        real_data_dir: str = "data/features/round_four",
        synthetic_data_dir: str = "data/synthetic",
        models_dir: str = "models",
        sequence_length: int = 100,
        gan_epochs: int = 1000,
        rl_episodes: int = 300
    ):
        """
        Initialize the integrated training pipeline.
        
        Args:
            products: List of product names to train agents for
            real_data_dir: Directory containing real feature data
            synthetic_data_dir: Directory to save synthetic data
            models_dir: Directory to save trained models
            sequence_length: Length of sequences for GAN
            gan_epochs: Number of epochs to train GAN
            rl_episodes: Number of episodes for RL training
        """
        self.products = products
        self.real_data_dir = Path(real_data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self.models_dir = Path(models_dir)
        self.sequence_length = sequence_length
        self.gan_epochs = gan_epochs
        self.rl_episodes = rl_episodes
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize containers
        self.gan_trainers: Dict[str, TimeSeriesTrainer] = {}
        self.rl_agents: Dict[str, RLTradingAgent] = {}
        self.trading_envs: Dict[str, TradingEnv] = {}
    
    def train_gan_for_product(self, product: str):
        """Train GAN model for a specific product."""
        print(f"\nTraining GAN for {product}")
        
        # Load real data
        data_path = self.real_data_dir / f"{product}_features.csv"
        df = pd.read_csv(data_path)
        
        # Get feature columns (excluding timestamp)
        feature_cols = [col for col in df.columns if col != 'timestamp']
        
        # Initialize trainer
        trainer = TimeSeriesTrainer(
            feature_dims=len(feature_cols),
            sequence_length=self.sequence_length
        )
        
        # Prepare and train
        dataset = trainer.prepare_data(df, feature_cols, 'timestamp')
        history = trainer.train(dataset, self.gan_epochs)
        
        # Save model and generate synthetic data
        model_dir = self.models_dir / f"{product}_gan"
        trainer.save_model(str(model_dir))
        trainer.save_synthetic_data(str(self.synthetic_data_dir))
        
        self.gan_trainers[product] = trainer
        
        return history
    
    def train_rl_agent_for_product(
        self,
        product: str,
        features: List[str],
        use_synthetic: bool = True,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        """Train RL agent for a specific product."""
        print(f"\nTraining RL agent for {product}")
        
        # Load real data
        real_data = pd.read_csv(self.real_data_dir / f"{product}_features.csv")
        
        # Load synthetic data if requested
        if use_synthetic:
            synthetic_data = pd.read_csv(self.synthetic_data_dir / f"{product}_synthetic_gan.csv")
            # Combine real and synthetic data
            data = pd.concat([real_data, synthetic_data], ignore_index=True)
            data = data.sort_values('timestamp')
        else:
            data = real_data
        
        # Initialize environment and agent
        env = TradingEnv(data, lags=1)
        agent = RLTradingAgent(features, alpha=alpha, gamma=gamma, epsilon=epsilon)
        
        # Training loop
        for ep in range(self.rl_episodes):
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
                    print(f"{product} | Episode {ep + 1}/{self.rl_episodes} | Reward: {total_reward:.3f}")
                    break
        
        # Save agent
        agent.save(str(self.models_dir / f"{product}_rl_agent.jsonpickle"))
        
        self.rl_agents[product] = agent
        self.trading_envs[product] = env
        
        return agent
    
    def train_all(self, features: List[str]):
        """Train GAN and RL agents for all products."""
        for product in self.products:
            # Train GAN
            gan_history = self.train_gan_for_product(product)
            
            # Train RL agent
            rl_agent = self.train_rl_agent_for_product(
                product,
                features,
                use_synthetic=True
            )
    
    def load_trained_models(self):
        """Load previously trained models."""
        for product in self.products:
            # Load GAN
            gan_path = self.models_dir / f"{product}_gan"
            if gan_path.exists():
                trainer = TimeSeriesTrainer(
                    feature_dims=len(self.get_feature_columns(product))
                )
                trainer.load_model(str(gan_path))
                self.gan_trainers[product] = trainer
            
            # Load RL agent
            rl_path = self.models_dir / f"{product}_rl_agent.jsonpickle"
            if rl_path.exists():
                agent = RLTradingAgent.load(str(rl_path))
                self.rl_agents[product] = agent
    
    def get_feature_columns(self, product: str) -> List[str]:
        """Get feature columns for a product."""
        df = pd.read_csv(self.real_data_dir / f"{product}_features.csv")
        return [col for col in df.columns if col != 'timestamp']
    
    def generate_synthetic_data(self, product: str, n_samples: int = 1000):
        """Generate synthetic data for a product."""
        if product not in self.gan_trainers:
            raise ValueError(f"No trained GAN found for {product}")
        
        trainer = self.gan_trainers[product]
        synthetic_sequences, _ = trainer.generate_samples(n_samples)
        
        # Convert to DataFrame
        feature_cols = self.get_feature_columns(product)
        synthetic_df = pd.DataFrame(
            synthetic_sequences.reshape(-1, len(feature_cols)),
            columns=feature_cols
        )
        
        # Add timestamps
        synthetic_df['timestamp'] = np.arange(len(synthetic_df)) * 100
        
        return synthetic_df
    
    def get_trading_action(self, product: str, state) -> int:
        """Get trading action from trained RL agent."""
        if product not in self.rl_agents:
            raise ValueError(f"No trained RL agent found for {product}")
        
        agent = self.rl_agents[product]
        phi = agent.featurize(state)
        return agent.choose_action(phi)

# Example usage
if __name__ == "__main__":
    # List of products to train agents for
    PRODUCTS = ["KELP", "SQUID_INK", "RAINFOREST_RESIN"]
    
    # Features for RL agents
    FEATURES = ["prediction", "r", "price"]
    
    # Initialize pipeline
    pipeline = IntegratedTrainingPipeline(
        products=PRODUCTS,
        gan_epochs=1000,
        rl_episodes=300
    )
    
    # Train all models
    pipeline.train_all(FEATURES)
    
    # Example: Generate synthetic data and get trading actions
    for product in PRODUCTS:
        # Generate synthetic data
        synthetic_data = pipeline.generate_synthetic_data(product)
        print(f"\nGenerated {len(synthetic_data)} synthetic samples for {product}")
        
        # Get trading action for last state
        last_state = synthetic_data.iloc[-1]
        action = pipeline.get_trading_action(product, last_state)
        print(f"Recommended action for {product}: {action}") 