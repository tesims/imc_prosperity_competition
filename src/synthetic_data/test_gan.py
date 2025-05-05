import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import FeatureDataLoader
from .train import TimeSeriesTrainer
from scipy import stats

def test_gan_pipeline(
    product: str = "CROISSANTS",
    sequence_length: int = 100,
    epochs: int = 1000,  # Reduced for testing
    validation_samples: int = 10
):
    """Test the complete GAN pipeline for a single product."""
    
    print("\n=== Testing GAN Pipeline ===")
    print(f"Product: {product}")
    print(f"Sequence Length: {sequence_length}")
    print(f"Epochs: {epochs}")
    
    # 1. Load and verify real data
    print("\n1. Loading real data...")
    loader = FeatureDataLoader()
    real_data = loader.get_product_features(product)
    print(f"Real data shape: {real_data.shape}")
    print("Sample features:")
    print(real_data.head())
    
    # 2. Initialize trainer
    print("\n2. Initializing trainer...")
    trainer = TimeSeriesTrainer(
        feature_loader=loader,
        product=product,
        sequence_length=sequence_length,
        epochs=epochs,
        validation_samples=validation_samples
    )
    
    # 3. Verify data preparation
    print("\n3. Verifying prepared data...")
    print(f"Sequence data shape: {trainer.real_data.shape}")
    print(f"Number of features: {trainer.feature_dims}")
    
    # 4. Train GAN
    print("\n4. Training GAN...")
    trainer.train()
    
    # 5. Generate and validate synthetic data
    print("\n5. Generating synthetic data...")
    output_dir = "data/synthetic/test"
    trainer.save_synthetic_data(output_dir)
    
    # 6. Load and verify synthetic data
    print("\n6. Verifying synthetic data...")
    synthetic_file = os.path.join(output_dir, f"{product}_synthetic_gan.csv")
    synthetic_data = pd.read_csv(synthetic_file)
    print("Synthetic data shape:", synthetic_data.shape)
    print("\nSynthetic data sample:")
    print(synthetic_data.head())
    
    # 7. Compare distributions
    print("\n7. Statistical comparison:")
    for column in synthetic_data.select_dtypes(include=[np.number]).columns:
        if column == 'timestamp':
            continue
        
        real_values = real_data[column].dropna().values
        synth_values = synthetic_data[column].dropna().values
        
        # Basic statistics
        print(f"\nFeature: {column}")
        print(f"{'Metric':<15} {'Real':<15} {'Synthetic':<15}")
        print("-" * 45)
        print(f"{'Mean':<15} {real_values.mean():<15.4f} {synth_values.mean():<15.4f}")
        print(f"{'Std':<15} {real_values.std():<15.4f} {synth_values.std():<15.4f}")
        print(f"{'Min':<15} {real_values.min():<15.4f} {synth_values.min():<15.4f}")
        print(f"{'Max':<15} {real_values.max():<15.4f} {synth_values.max():<15.4f}")
        
        # KS test
        statistic, pvalue = stats.ks_2samp(real_values, synth_values)
        print(f"KS test p-value: {pvalue:.4f}")
    
    # 8. Plot comparisons
    print("\n8. Plotting comparisons...")
    trainer.plot_training_history()
    trainer.plot_sample_comparison()
    
    return trainer, synthetic_data

if __name__ == "__main__":
    # Test with reduced epochs for faster verification
    trainer, synthetic_data = test_gan_pipeline(
        product="CROISSANTS",
        epochs=1000,
        sequence_length=100
    ) 