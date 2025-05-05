# IMC Prosperity Trading System

A complete pipeline for processing L2 orderbook data, generating synthetic data using GANs, and training RL trading agents.

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw L2 orderbook CSV files
│   ├── processed/        # Processed orderbook data by product
│   ├── features/         # Feature-engineered data
│   └── synthetic/        # GAN-generated synthetic data
├── models/              # Trained models (GAN and RL agents)
├── src/
│   ├── orderbook_features/  # Feature engineering module
│   ├── synthetic_data/      # GAN implementation
│   ├── rl_agents/          # RL trading agents
│   └── pipeline.py         # Main pipeline script
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imc-prosperity.git
cd imc-prosperity
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your L2 orderbook data files in the `data/raw/` directory. Files should be CSV format with columns:
   - timestamp
   - product
   - bid_price_1, bid_size_1, ask_price_1, ask_size_1, etc.

2. Run the complete pipeline:
```bash
python -m src.pipeline
```

This will:
- Discover available products in your data
- Process raw orderbook data into features
- Generate synthetic data using GANs
- Train RL agents for each product

## Pipeline Steps

1. **Data Processing**
   - Processes raw L2 orderbook data
   - Calculates features like price, volume, volatility
   - Organizes data by product

2. **Synthetic Data Generation**
   - Uses GANs to generate synthetic market data
   - Helps augment training data for RL agents
   - Preserves market characteristics

3. **RL Agent Training**
   - Trains product-specific trading agents
   - Uses both real and synthetic data
   - Implements sophisticated reward function
   - Manages positions and risk

## Configuration

Key parameters can be adjusted in `src/pipeline.py`:

```python
pipeline = TradingPipeline(
    raw_data_dir="data/raw",
    processed_data_dir="data/processed",
    features_dir="data/features",
    synthetic_dir="data/synthetic",
    models_dir="models",
    gan_epochs=1000,    # Number of GAN training epochs
    rl_episodes=300     # Number of RL training episodes
)
```

## Output

The pipeline produces:

1. **Processed Data**
   - Feature-engineered data for each product
   - Combined real and synthetic data

2. **Trained Models**
   - GAN models for each product
   - RL trading agents for each product

3. **Performance Metrics**
   - Training rewards and Sharpe ratios
   - Position and PnL statistics

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
