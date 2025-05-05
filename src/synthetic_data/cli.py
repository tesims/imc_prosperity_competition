import argparse
import os
from data import FeatureDataLoader
from .train import TimeSeriesTrainer

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic trading data using GAN')
    parser.add_argument('--product', type=str, required=True,
                       help='Product to generate data for')
    parser.add_argument('--output-dir', type=str, default='data/synthetic',
                       help='Directory to save synthetic data')
    parser.add_argument('--sequence-length', type=int, default=100,
                       help='Length of sequences to generate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5000,
                       help='Number of training epochs')
    parser.add_argument('--validation-samples', type=int, default=25,
                       help='Number of synthetic samples for validation')
    
    args = parser.parse_args()
    
    # Initialize feature loader
    feature_loader = FeatureDataLoader()
    
    # Initialize and train GAN
    trainer = TimeSeriesTrainer(
        feature_loader=feature_loader,
        product=args.product,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_samples=args.validation_samples
    )
    
    # Train the model
    trainer.train()
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_sample_comparison()
    
    # Save synthetic data
    trainer.save_synthetic_data(args.output_dir)

if __name__ == "__main__":
    main()
