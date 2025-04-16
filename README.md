# multi-agent-reinforcement-learning

This project develops a multi-asset trading framework combining market microstructure analysis with reinforcement learning. 

Starting with L2 order book data, I engineer features capturing both traditional technical indicators (MACD, RSI) and microstructure dynamics (order flow imbalance, volume-weighted spreads). 

To address data limitations, I implement a conditional GAN architecture with spectral normalization, trained to preserve key statistical properties including autocorrelation structures and fat-tailed return distributions. 

Individual trading agents for each asset class are optimized using proximal policy optimization (PPO), with reward functions incorporating both Sharpe ratio and drawdown constraints. 

The multi-agent coordination problem is formulated as a partially observable Markov game, where a meta-controller performs risk-aware capital allocation across strategies. 

While computational constraints limit the scale of backtesting, the system demonstrates statistically significant outperformance versus baseline strategies in controlled experiments, particularly in regimes with elevated volatility.  This work provides a practical demonstration of how modern machine learning techniques can be adapted for quantitative trading while maintaining computational feasibility on limited hardware.

## Process Overview

### 1. Data Cleaning & Feature Engineering

**Input**: Raw L2 order book data (bid/ask prices, volumes, PnL) for each product (stocks, ETFs, etc.)

**Processing**:
- **Cleaning**:
  - Handle missing values
  - Remove outliers
  - Align timestamps

- **Feature Calculation** (per product):
  *Classic Indicators*:
  - Rolling volatility
  - RSI, MACD, CCI
  - Z-score normalization

  *Order Book Dynamics*:
  - Depth volatility
  - Bid-ask spread
  - Order book imbalance

  *Volume Metrics*:
  - Volume-weighted mid-price
  - Cumulative volume
  - Trade volume proxy

  *Normalized Returns*:
  - Mid-price returns
  - PnL derivatives

### 2. Synthetic Data Generation (GAN)

**Input**: Original data + engineered features (limited historical samples)

**GAN Architecture**:
- Wasserstein GAN with Gradient Penalty (WGAN-GP)
- Preserves:
  - Temporal dependencies (volatility clustering)
  - Microstructure patterns (order book dynamics)
  - Volume correlations

**Output**: Augmented dataset (real + synthetic) for RL training

### 3. Reinforcement Learning Agent Training

**Agent Selection**:
- Backtest on synthetic data (Sharpe ratio comparison)
- Algorithm assignment:
  - PPO for high-frequency assets
  - SAC for multi-modal spaces

**Training Framework**:
- *State Space*: Normalized features + portfolio context
- *Reward Function*: Sharpe ratio + drawdown penalties
- *Environment*: Synthetic market with stochastic slippage

### 4. Multi-Agent Portfolio Optimization

**Coordination**:
- Hierarchical RL architecture
- Meta-controller for risk-parity allocation
- Product-specific agent policies

**Optimization Targets**:
- Portfolio CAGR
- Sortino ratio
- Maximum drawdown constraints

**Implementation**:
- Shared critic network
- Auction-based rebalancing mechanism
