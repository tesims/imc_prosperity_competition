import argparse
import os
from .generator import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic trading data')
    parser.add_argument('--output-dir', type=str, default='data/synthetic',
                       help='Directory to save synthetic data')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples to generate per product')
    parser.add_argument('--product', type=str,
                       help='Specific product to generate data for (optional)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    if args.product:
        # Generate data for specific product
        print(f"Generating {args.n_samples} samples for {args.product}")
        df = generator.generate_synthetic_data(args.product, args.n_samples)
        output_file = os.path.join(args.output_dir, f"{args.product}_synthetic.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved synthetic data to {output_file}")
    else:
        # Generate data for all products
        print(f"Generating {args.n_samples} samples for each product")
        synthetic_data = generator.generate_all_products(args.n_samples)
        for product, df in synthetic_data.items():
            output_file = os.path.join(args.output_dir, f"{product}_synthetic.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved synthetic data for {product} to {output_file}")

if __name__ == "__main__":
    main()
