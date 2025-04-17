import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from data import FeatureDataLoader

class SyntheticDataGenerator:
    def __init__(self, feature_loader: Optional[FeatureDataLoader] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            feature_loader: FeatureDataLoader instance to use for getting real data statistics
        """
        self.feature_loader = feature_loader or FeatureDataLoader()
        self.products = self.feature_loader.get_available_products()
        self.feature_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._compute_feature_statistics()
    
    def _compute_feature_statistics(self):
        """Compute statistics for each product's features."""
        for product in self.products:
            self.feature_stats[product] = self.feature_loader.get_feature_statistics(product)
    
    def generate_synthetic_data(self, product: str, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data for a product based on its feature statistics.
        
        Args:
            product: Name of the product to generate data for
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame containing synthetic data
        """
        if product not in self.feature_stats:
            raise ValueError(f"No statistics available for product {product}")
            
        stats = self.feature_stats[product]
        synthetic_data = {}
        
        # Generate synthetic values for each numeric feature
        for feature, feature_stats in stats.items():
            mean = feature_stats['mean']
            std = feature_stats['std']
            
            # Generate normally distributed values based on feature statistics
            synthetic_values = np.random.normal(mean, std, n_samples)
            
            # Clip values to min/max range from real data
            synthetic_values = np.clip(
                synthetic_values,
                feature_stats['min'],
                feature_stats['max']
            )
            
            synthetic_data[feature] = synthetic_values
        
        # Create timestamps
        base_timestamp = 0
        step = 100  # 100ms steps
        synthetic_data['timestamp'] = [base_timestamp + i * step for i in range(n_samples)]
        
        return pd.DataFrame(synthetic_data)
    
    def generate_all_products(self, n_samples: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for all available products.
        
        Args:
            n_samples: Number of samples to generate per product
            
        Returns:
            Dictionary mapping product names to their synthetic DataFrames
        """
        return {
            product: self.generate_synthetic_data(product, n_samples)
            for product in self.products
        } 