import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime
import logging
import torch

from src.orderbook_features.feature_calculator import OrderBookFeatureCalculator
from src.synthetic_data.models import TimeGAN
from src.synthetic_data.train import TimeSeriesTrainer
from src.rl_agents.enhanced_rl_agent import EnhancedRLTradingAgent
from src.rl_agents.enhanced_env import EnhancedTradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingPipeline:
    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        features_dir: str = "data/features",
        synthetic_dir: str = "data/synthetic",
        models_dir: str = "models",
        gan_epochs: int = 1000,
        rl_episodes: int = 300,
        sequence_length: int = 50
    ):
        # Initialize directories
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.features_dir = Path(features_dir)
        self.synthetic_dir = Path(synthetic_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories if they don't exist
        for dir_path in [self.raw_data_dir, self.processed_data_dir, 
                        self.features_dir, self.synthetic_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.gan_epochs = gan_epochs
        self.rl_episodes = rl_episodes
        self.sequence_length = sequence_length
        
        # Initialize components
        self.feature_calculator = OrderBookFeatureCalculator()
    
    def discover_products(self) -> List[str]:
        """Discover unique products from raw data files."""
        products = set()
        for file in self.raw_data_dir.glob("*.csv"):
            df = pd.read_csv(file)
            if "product" in df.columns:
                products.update(df["product"].unique())
            elif "symbol" in df.columns:
                products.update(df["symbol"].unique())
        return sorted(list(products))
    
    def process_raw_data(self, product: str) -> pd.DataFrame:
        """Process raw data for a specific product."""
        logger.info(f"Processing raw data for {product}")
        
        # Collect all data for this product
        dfs = []
        for file in self.raw_data_dir.glob("*.csv"):
            df = pd.read_csv(file)
            # Support both 'symbol' and 'product' column names
            if "symbol" in df.columns:
                key_col = "symbol"
            elif "product" in df.columns:
                key_col = "product"
            else:
                continue
            product_data = df[df[key_col] == product].copy()
            
            if not product_data.empty:
                # Transform the data into L2 orderbook format
                timestamps = sorted(product_data["timestamp"].unique())
                transformed_data = []
                
                for ts in timestamps:
                    ts_data = product_data[product_data["timestamp"] == ts]
                    bids = ts_data[ts_data["side"] == "BID"].sort_values("price", ascending=False)
                    asks = ts_data[ts_data["side"] == "ASK"].sort_values("price")
                    
                    row = {
                        "timestamp": ts,
                        "product": product
                    }
                    
                    # Add bid levels
                    for i, bid in enumerate(bids.itertuples(), 1):
                        row[f"bid_price_{i}"] = bid.price
                        row[f"bid_size_{i}"] = bid.quantity
                        
                    # Add ask levels
                    for i, ask in enumerate(asks.itertuples(), 1):
                        row[f"ask_price_{i}"] = ask.price
                        row[f"ask_size_{i}"] = ask.quantity
                        
                    transformed_data.append(row)
                
                transformed_df = pd.DataFrame(transformed_data)
                dfs.append(transformed_df)
        
        if not dfs:
            raise ValueError(f"No data found for product {product}")
        
        # Combine and sort by timestamp
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.sort_values("timestamp", inplace=True)
        
        # Save processed data
        output_path = self.processed_data_dir / f"{product}_processed.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return combined_df
    
    def calculate_features(self, product: str, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for a product using the OrderBookFeatureCalculator."""
        logger.info(f"Calculating features for {product}")
        
        features_df = self.feature_calculator.calculate_features(df)
        
        # Save features
        output_path = self.features_dir / f"{product}_features.csv"
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
        
        return features_df
    
    def prepare_sequences(self, features: Union[pd.DataFrame, torch.Tensor], sequence_length: int) -> Optional[torch.Tensor]:
        """
        Prepare sequences for GAN training from features DataFrame or Tensor.
        
        Args:
            features: DataFrame or Tensor containing feature data
            sequence_length: Length of sequences to create
            
        Returns:
            torch.Tensor of shape (n_sequences, sequence_length, n_features) or None if not enough data
        """
        # Convert features to numpy array if it's a DataFrame
        if isinstance(features, pd.DataFrame):
            numeric_features = features.select_dtypes(include=[np.number])
            if numeric_features.empty:
                logger.warning("No numeric features found in data")
                return None
            features_array = numeric_features.astype(np.float32).values
        else:
            features_array = features.numpy() if isinstance(features, torch.Tensor) else features
        
        # Create sequences using sliding window
        n_samples = len(features_array)
        n_features = features_array.shape[1]
        
        # If not enough samples, use a smaller sequence length
        if n_samples < sequence_length:
            logger.warning(f"Not enough samples ({n_samples}) for sequence length {sequence_length}. "
                          f"Using smaller sequence length: {max(5, n_samples//2)}")
            sequence_length = max(5, n_samples//2)  # Use at least 5 or half the samples
            
            # Still not enough samples
            if n_samples < 10:  # Need at least 10 samples to create meaningful sequences
                logger.error(f"Too few samples ({n_samples}), need at least 10")
                return None
        
        # Pre-allocate numpy array for sequences
        n_sequences = n_samples - sequence_length + 1
        sequences = np.zeros((n_sequences, sequence_length, n_features), dtype=np.float32)
        
        # Fill sequences array
        for i in range(n_sequences):
            sequences[i] = features_array[i:i + sequence_length]
        
        # Convert to tensor and ensure contiguous memory layout
        sequences_tensor = torch.from_numpy(sequences).float().contiguous()
        
        logger.info(f"Created {n_sequences} sequences with shape: {sequences_tensor.shape}")
        return sequences_tensor
    
    def train_gan(self, product: str, features_df: pd.DataFrame) -> None:
        """
        Train GAN for a specific product using its features.
        
        Args:
            product: Name of the product
            features_df: DataFrame containing feature data
        """
        logger.info(f"Training GAN for {product}")
        logger.info(f"Using features: {features_df.columns}")
        
        try:
            # Create synthetic data directory if it doesn't exist
            synthetic_dir = Path("data/synthetic/round_four")
            synthetic_dir.mkdir(parents=True, exist_ok=True)
            
            # Get numeric columns for feature dimensions
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 5:  # Minimum required features
                raise ValueError(f"Not enough numeric features: {len(numeric_cols)}")
            
            # Prepare sequences for training
            sequences = self.prepare_sequences(features_df[numeric_cols], sequence_length=self.sequence_length)
            if sequences is None:
                logger.error(f"Could not create valid sequences for {product}")
                return
                
            # Ensure we have enough sequences for training
            if sequences.shape[0] < 10:  # Minimum required sequences
                logger.error(f"Not enough sequences for {product}: {sequences.shape[0]}")
                return
            
            # Use adaptive sequence length from prepare_sequences
            actual_seq_length = sequences.shape[1]
            
            # Initialize and train GAN trainer
            trainer = TimeSeriesTrainer(
                feature_dims=len(numeric_cols),
                seq_length=actual_seq_length,
                latent_dims=64,
                gradient_penalty_weight=10.0,
                lr=0.0002,
                beta1=0.5,
                beta2=0.9
            )
            
            # Train the GAN
            trainer.train(sequences, n_epochs=self.gan_epochs)
            
            # Save the trained model
            model_path = os.path.join(self.models_dir, f"{product}_gan.pt")
            trainer.save_checkpoint(model_path)
            
            # Generate synthetic data
            n_synthetic = len(features_df)
            synthetic_df = trainer.generate_samples(n_samples=n_synthetic)
            
            # Add non-numeric columns from original data
            for col in features_df.columns:
                if col not in numeric_cols:
                    if col == 'day':
                        synthetic_df[col] = features_df[col].max() + 1
                    elif col == 'timestamp':
                        synthetic_df[col] = pd.date_range(
                            start=features_df[col].max(),
                            periods=len(synthetic_df),
                            freq='1min'
                        )
                    else:
                        synthetic_df[col] = features_df[col].iloc[0]
            
            # Save synthetic data
            output_path = os.path.join(synthetic_dir, f"{product}_synthetic.csv")
            synthetic_df.to_csv(output_path, index=False)
            
            # Train RL agent using both real and synthetic data
            agent = self.train_rl_agent(product, features_df, synthetic_df)
            
            # Save RL agent
            agent_path = os.path.join(self.models_dir, f"{product}_rl_agent.pt")
            agent.save(str(agent_path))
            logger.info(f"Saved RL agent to {agent_path}")
            
        except Exception as e:
            logger.error(f"Error processing {product}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return
    
    def train_rl_agent(self, product: str, features_df: pd.DataFrame, 
                       synthetic_df: Optional[pd.DataFrame] = None) -> EnhancedRLTradingAgent:
        """Train an RL agent using both real and synthetic data."""
        logger.info(f"Training RL agent for {product}")
        
        # Combine real and synthetic data if available
        if synthetic_df is not None:
            training_data = pd.concat([features_df, synthetic_df], ignore_index=True)
            training_data = training_data.sort_values('timestamp')
        else:
            training_data = features_df
        
        # Initialize environment and agent
        env = EnhancedTradingEnv(
            df=training_data,
            position_limit=100,
            transaction_cost=0.001,
            history_window=self.sequence_length  # Use same window as GAN
        )
        
        agent = EnhancedRLTradingAgent(
            state_size=env.observation_space.shape[0],
            action_size=5,
            alpha=0.001,
            gamma=0.99,
            epsilon=1.0
        )
        
        # Train agent
        total_rewards = []
        for episode in range(self.rl_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{self.rl_episodes}, "
                          f"Average Reward: {np.mean(total_rewards[-10:]):.2f}")
        
        # Save agent
        agent_path = self.models_dir / f"{product}_rl_agent.pt"
        agent.save(str(agent_path))
        logger.info(f"Saved RL agent to {agent_path}")
        
        return agent
    
    def run(self):
        """Run the complete pipeline."""
        logger.info("Starting pipeline execution")
        
        # Get products from feature files instead of raw data
        products = []
        for file in self.features_dir.glob("*_features.csv"):
            product = file.stem.replace("_features", "")
            products.append(product)
        
        logger.info(f"Found products with features: {products}")
        
        for product in products:
            try:
                # Load existing features
                feature_file = self.features_dir / f"{product}_features.csv"
                features_df = pd.read_csv(feature_file)
                
                # Add timestamp if not present
                if 'timestamp' not in features_df.columns:
                    features_df['timestamp'] = pd.date_range(
                        start='2024-01-01',
                        periods=len(features_df),
                        freq='1min'
                    )
                
                # Train GAN and generate synthetic data
                self.train_gan(product, features_df)
                synthetic_df = pd.read_csv(self.synthetic_dir / f"{product}_synthetic.csv")
                
                # Train RL agent
                agent = self.train_rl_agent(product, features_df, synthetic_df)
                
                logger.info(f"Successfully completed pipeline for {product}")
                
            except Exception as e:
                logger.error(f"Error processing {product}: {str(e)}")
                continue
        
        logger.info("Pipeline execution completed")

if __name__ == "__main__":
    pipeline = TradingPipeline()
    pipeline.run() 