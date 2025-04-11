from datamodel import Listing, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from algorithms.example import Trader
import pandas as pd
import json

class RealDataSimulation:
    def __init__(self, prices_file):
        self.running = True
        self.current_timestamp = 0
        self.prices_df = pd.read_csv(prices_file, sep=';')
        self.timestamps = sorted(self.prices_df['timestamp'].unique())
        self.current_time_idx = 0
        self.trader_data = ""
        
    def is_running(self):
        return self.current_time_idx < len(self.timestamps)
    
    def create_order_depth(self, product_data):
        """Create OrderDepth from price data row"""
        order_depth = OrderDepth()
        
        # Add bid orders
        if not pd.isna(product_data['bid_price_1']):
            order_depth.buy_orders[product_data['bid_price_1']] = product_data['bid_volume_1']
        if not pd.isna(product_data['bid_price_2']):
            order_depth.buy_orders[product_data['bid_price_2']] = product_data['bid_volume_2']
        if not pd.isna(product_data['bid_price_3']):
            order_depth.buy_orders[product_data['bid_price_3']] = product_data['bid_volume_3']
            
        # Add ask orders
        if not pd.isna(product_data['ask_price_1']):
            order_depth.sell_orders[product_data['ask_price_1']] = product_data['ask_volume_1']
        if not pd.isna(product_data['ask_price_2']):
            order_depth.sell_orders[product_data['ask_price_2']] = product_data['ask_volume_2']
        if not pd.isna(product_data['ask_price_3']):
            order_depth.sell_orders[product_data['ask_price_3']] = product_data['ask_volume_3']
            
        return order_depth
    
    def get_current_state(self):
        """Create TradingState from current timestamp data"""
        current_time = self.timestamps[self.current_time_idx]
        current_data = self.prices_df[self.prices_df['timestamp'] == current_time]
        
        order_depths = {}
        for _, row in current_data.iterrows():
            product = row['product']
            order_depths[product] = self.create_order_depth(row)
        
        return TradingState(
            timestamp=current_time,
            listings={p: Listing(p, p, "USD") for p in order_depths.keys()},
            order_depths=order_depths,
            own_trades=[],
            market_trades=[],
            position={p: 0 for p in order_depths.keys()},
            observations={},
            traderData=self.trader_data
        )
    
    def submit_orders(self, orders):
        """Process the submitted orders and update positions"""
        current_time = self.timestamps[self.current_time_idx]
        current_data = self.prices_df[self.prices_df['timestamp'] == current_time]
        
        for product, product_orders in orders.items():
            product_data = current_data[current_data['product'] == product].iloc[0]
            
            for order in product_orders:
                print(f"Order: {order.quantity}x {product} @ {order.price}")
                
                # Simple simulation of order execution
                if order.quantity > 0:  # Buy order
                    if order.price >= product_data['ask_price_1']:
                        print(f"  Executed at {product_data['ask_price_1']}")
                elif order.quantity < 0:  # Sell order
                    if order.price <= product_data['bid_price_1']:
                        print(f"  Executed at {product_data['bid_price_1']}")
    
    def process_conversions(self, conversions):
        pass
    
    def update_trader_data(self, trader_data):
        self.trader_data = trader_data
    
    def next_iteration(self):
        self.current_time_idx += 1

def main():
    # Initialize simulation with real price data
    prices_file = "data/prices/round_one/prices_round_1_day_0.csv"
    simulation = RealDataSimulation(prices_file)
    
    # Create trader instance
    trader = Trader()
    
    print("Starting simulation with real market data...")
    print("==========================================")
    
    # Run simulation
    while simulation.is_running():
        # Get current market state
        state = simulation.get_current_state()
        
        # Get trader's actions
        result, conversions, trader_data = trader.run(state)
        
        # Process actions
        simulation.submit_orders(result)
        simulation.process_conversions(conversions)
        simulation.update_trader_data(trader_data)
        
        # Move to next timestamp
        simulation.next_iteration()
    
    print("\nSimulation completed!")
    print("====================")

if __name__ == "__main__":
    main() 