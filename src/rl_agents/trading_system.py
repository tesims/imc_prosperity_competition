import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from integrated_training import IntegratedTrainingPipeline

class MultiAgentTradingSystem:
    def __init__(
        self,
        products: List[str],
        models_dir: str = "models",
        position_limits: Optional[Dict[str, int]] = None
    ):
        """
        Initialize multi-agent trading system.
        
        Args:
            products: List of products to trade
            models_dir: Directory containing trained models
            position_limits: Optional position limits per product
        """
        self.products = products
        self.models_dir = Path(models_dir)
        self.position_limits = position_limits or {p: 50 for p in products}
        
        # Load trained models
        self.pipeline = IntegratedTrainingPipeline(products)
        self.pipeline.load_trained_models()
        
        # Initialize trading state
        self.positions = defaultdict(int)
        self.trades_history = defaultdict(list)
        self.pnl = defaultdict(float)
    
    def process_market_update(
        self,
        product: str,
        state: pd.Series,
        current_price: float
    ) -> Tuple[str, int]:
        """
        Process market update and get trading decision.
        
        Args:
            product: Product name
            state: Current market state
            current_price: Current market price
            
        Returns:
            Tuple of (action_type, quantity)
        """
        # Get action from RL agent
        action = self.pipeline.get_trading_action(product, state)
        
        # Convert action to trade decision
        current_position = self.positions[product]
        position_limit = self.position_limits[product]
        
        # Define action mapping (example)
        # 0: Strong Sell, 1: Weak Sell, 2: Hold, 3: Weak Buy, 4: Strong Buy
        quantity = 0
        action_type = "HOLD"
        
        if action == 0:  # Strong Sell
            if current_position > -position_limit:
                quantity = min(10, current_position + position_limit)
                action_type = "SELL"
        elif action == 1:  # Weak Sell
            if current_position > -position_limit:
                quantity = min(5, current_position + position_limit)
                action_type = "SELL"
        elif action == 3:  # Weak Buy
            if current_position < position_limit:
                quantity = min(5, position_limit - current_position)
                action_type = "BUY"
        elif action == 4:  # Strong Buy
            if current_position < position_limit:
                quantity = min(10, position_limit - current_position)
                action_type = "BUY"
        
        # Update position if trade is executed
        if quantity > 0:
            if action_type == "BUY":
                self.positions[product] += quantity
                self.trades_history[product].append({
                    'type': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': state.get('timestamp', None)
                })
            else:  # SELL
                self.positions[product] -= quantity
                self.trades_history[product].append({
                    'type': 'SELL',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': state.get('timestamp', None)
                })
        
        return action_type, quantity
    
    def update_pnl(self, product: str, current_price: float):
        """Update PnL for a product."""
        position = self.positions[product]
        trades = self.trades_history[product]
        
        if not trades:
            return
        
        # Calculate realized PnL from closed trades
        realized_pnl = 0
        for i in range(len(trades) - 1):
            trade = trades[i]
            next_trade = trades[i + 1]
            
            if trade['type'] == 'BUY' and next_trade['type'] == 'SELL':
                realized_pnl += (next_trade['price'] - trade['price']) * min(trade['quantity'], next_trade['quantity'])
            elif trade['type'] == 'SELL' and next_trade['type'] == 'BUY':
                realized_pnl += (trade['price'] - next_trade['price']) * min(trade['quantity'], next_trade['quantity'])
        
        # Calculate unrealized PnL from current position
        unrealized_pnl = 0
        if position != 0:
            last_trade = trades[-1]
            if last_trade['type'] == 'BUY':
                unrealized_pnl = (current_price - last_trade['price']) * position
            else:  # SELL
                unrealized_pnl = (last_trade['price'] - current_price) * abs(position)
        
        self.pnl[product] = realized_pnl + unrealized_pnl
    
    def get_position(self, product: str) -> int:
        """Get current position for a product."""
        return self.positions[product]
    
    def get_pnl(self, product: str) -> float:
        """Get current PnL for a product."""
        return self.pnl[product]
    
    def get_trade_history(self, product: str) -> List[Dict]:
        """Get trade history for a product."""
        return self.trades_history[product]
    
    def generate_trading_signals(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Generate trading signals for all products.
        
        Args:
            market_data: Dictionary of market data for each product
            
        Returns:
            Dictionary of trading signals for each product
        """
        signals = {}
        
        for product in self.products:
            if product not in market_data:
                continue
            
            data = market_data[product]
            product_signals = []
            
            for _, row in data.iterrows():
                action_type, quantity = self.process_market_update(
                    product,
                    row,
                    row['price']  # Assuming 'price' column exists
                )
                
                if quantity > 0:  # Only include non-HOLD signals
                    product_signals.append((action_type, quantity))
                
                # Update PnL
                self.update_pnl(product, row['price'])
            
            signals[product] = product_signals
        
        return signals

# Example usage
if __name__ == "__main__":
    # List of products
    PRODUCTS = ["KELP", "SQUID_INK", "RAINFOREST_RESIN"]
    
    # Initialize trading system
    trading_system = MultiAgentTradingSystem(
        products=PRODUCTS,
        position_limits={"KELP": 50, "SQUID_INK": 50, "RAINFOREST_RESIN": 50}
    )
    
    # Example market data (you would replace this with real data)
    market_data = {
        product: pd.DataFrame({
            'timestamp': range(100),
            'price': np.random.normal(100, 10, 100),
            'volume': np.random.randint(1, 100, 100),
            'prediction': np.random.normal(0, 1, 100),
            'r': np.random.normal(0, 1, 100)
        }) for product in PRODUCTS
    }
    
    # Generate trading signals
    signals = trading_system.generate_trading_signals(market_data)
    
    # Print results
    for product in PRODUCTS:
        print(f"\nResults for {product}:")
        print(f"Number of trades: {len(trading_system.get_trade_history(product))}")
        print(f"Final position: {trading_system.get_position(product)}")
        print(f"Final PnL: {trading_system.get_pnl(product):.2f}")
        print("Sample of trading signals:", signals[product][:5]) 