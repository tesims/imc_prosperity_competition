import pandas as pd
import numpy as np
from typing import List, Dict

class OrderBookFeatureCalculator:
    """Calculate features from L2 orderbook data."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from orderbook data."""
        features = df.copy()
        
        # Ensure numeric columns
        numeric_columns = ['bid_price_1', 'bid_size_1', 'ask_price_1', 'ask_size_1']
        for col in numeric_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        
        # Calculate mid price
        features['mid_price'] = (
            features['bid_price_1'].fillna(0) + 
            features['ask_price_1'].fillna(0)
        ) / 2
        
        # Calculate spread
        features['spread'] = features['ask_price_1'].fillna(0) - features['bid_price_1'].fillna(0)
        
        # Calculate imbalance
        bid_volume = features['bid_size_1'].fillna(0)
        ask_volume = features['ask_size_1'].fillna(0)
        total_volume = bid_volume + ask_volume
        features['imbalance'] = (bid_volume - ask_volume) / total_volume.where(total_volume != 0, 0)
        
        # Calculate price changes
        features['price_change'] = features['mid_price'].diff()
        
        # Calculate rolling statistics
        features['volatility'] = features['mid_price'].rolling(self.window_size).std()
        features['momentum'] = features['price_change'].rolling(self.window_size).mean()
        features['volume_trend'] = total_volume.rolling(self.window_size).mean()
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Drop non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        return numeric_features 