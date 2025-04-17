import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class FeatureDataLoader:
    def __init__(self, feature_dir: str = "data/features/round_four"):
        """
        Initialize the feature data loader.
        
        Args:
            feature_dir: Directory containing the feature CSV files
        """
        self.feature_dir = feature_dir
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self._load_all_features()
    
    def _load_all_features(self):
        """Load all feature files from the feature directory."""
        print(f"Loading features from {self.feature_dir}")
        
        for file in os.listdir(self.feature_dir):
            if file.endswith("_features.csv"):
                product = file.replace("_features.csv", "")
                file_path = os.path.join(self.feature_dir, file)
                print(f"Loading features for {product}")
                
                try:
                    df = pd.read_csv(file_path)
                    self.data_cache[product] = df
                    print(f"Loaded {len(df)} rows for {product}")
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
    
    def get_product_features(self, product: str) -> Optional[pd.DataFrame]:
        """
        Get features for a specific product.
        
        Args:
            product: Name of the product (e.g., 'CROISSANTS')
            
        Returns:
            DataFrame containing the features or None if product not found
        """
        return self.data_cache.get(product)
    
    def get_available_products(self) -> List[str]:
        """Get list of products with available feature data."""
        return list(self.data_cache.keys())
    
    def get_feature_names(self) -> List[str]:
        """Get list of available feature names."""
        if not self.data_cache:
            return []
        # Get feature names from first available product
        first_product = next(iter(self.data_cache.values()))
        return list(first_product.columns)
    
    def get_time_window(self, product: str, start_timestamp: int, end_timestamp: int) -> Optional[pd.DataFrame]:
        """
        Get features for a specific product within a time window.
        
        Args:
            product: Name of the product
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            
        Returns:
            DataFrame containing the features within the time window or None if not found
        """
        df = self.get_product_features(product)
        if df is None:
            return None
            
        mask = (df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)
        return df[mask]
    
    def get_latest_features(self, product: str, n_rows: int = 1) -> Optional[pd.DataFrame]:
        """
        Get the most recent features for a product.
        
        Args:
            product: Name of the product
            n_rows: Number of most recent rows to return
            
        Returns:
            DataFrame containing the most recent features or None if not found
        """
        df = self.get_product_features(product)
        if df is None:
            return None
            
        return df.tail(n_rows)
    
    def get_feature_statistics(self, product: str) -> Dict[str, Dict[str, float]]:
        """
        Get basic statistics for all features of a product.
        
        Args:
            product: Name of the product
            
        Returns:
            Dictionary containing statistics for each feature
        """
        df = self.get_product_features(product)
        if df is None:
            return {}
            
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
            
        return stats

# Example usage:
if __name__ == "__main__":
    # Initialize the data loader
    loader = FeatureDataLoader()
    
    # Print available products
    print("\nAvailable products:")
    print(loader.get_available_products())
    
    # Print available features
    print("\nAvailable features:")
    print(loader.get_feature_names())
    
    # Example: Get features for CROISSANTS
    croissants_df = loader.get_product_features("CROISSANTS")
    if croissants_df is not None:
        print("\nCROISSANTS features shape:", croissants_df.shape)
        
        # Get latest features
        latest = loader.get_latest_features("CROISSANTS", n_rows=5)
        print("\nLatest CROISSANTS features:")
        print(latest[['timestamp', 'mid_price', 'volume_weighted_return', 'rsi']]) 