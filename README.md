# multi-agent-reinforcement-learning

Process Overview
Data Cleaning & Feature Engineering

Input: Raw L2 order book data (bid/ask prices, volumes, PnL) for each product (stocks, ETFs, etc.).

Cleaning: Handle missing values, outliers, and timestamp alignment.

Feature Calculation: For each product, compute:

Classic Indicators: Rolling volatility, RSI, MACD, CCI, Z-score.

Order Book Dynamics: Depth volatility, bid-ask spread, order book imbalance.

Volume Metrics: Volume-weighted mid-price, cumulative volume, trade volume proxy.

Normalized Returns: Mid-price returns, PnL derivatives.

Synthetic Data Generation (GAN)

Input: Original data + engineered features (limited historical samples).

GAN Training: A Wasserstein GAN (WGAN-GP) learns the joint distribution of raw and derived features to generate indistinguishable synthetic market data, preserving:

Temporal dependencies (e.g., volatility clustering).

Microstructure patterns (order book dynamics, volume correlations).

Output: Augmented dataset (real + synthetic) for robust RL training.

Reinforcement Learning Agent Selection & Training

Per-Product Analysis: Evaluate synthetic data performance (e.g., backtest Sharpe ratio) to select the best RL algorithm (PPO for high-frequency, SAC for multi-modal spaces).

Agent Training: Each product’s agent learns via:

State Space: Normalized features + portfolio context (e.g., inventory risk).

Reward: Risk-adjusted returns (e.g., Sharpe + penalty for drawdowns).

Environment: Synthetic market simulator with stochastic slippage.

Multi-Agent Portfolio Optimization

Framework: Hierarchical RL or Markov Game to coordinate agents:

Meta-controller allocates capital based on correlation-aware risk parity.

Individual agents execute product-specific strategies.

Optimization: Maximize portfolio-wide metrics (e.g., CAGR, Sortino ratio) via shared critic or auction-based rebalancing.
