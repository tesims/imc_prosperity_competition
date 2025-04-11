from datamodel import Listing, Observation, Order, OrderDepth, TradingState, Trade
from algorithms.example import Trader
import json

class SimpleSimulationEnvironment:
    def __init__(self):
        self.running = True
        self.current_timestamp = 0
        self.iteration = 0
        self.max_iterations = 10  # Run 10 iterations by default
        
        # Position limits for each product
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        # Initialize price history for each product with characteristics mentioned
        self.price_history = {
            "RAINFOREST_RESIN": {"last_price": 100, "volatility": 0.5},  # Stable value
            "KELP": {"last_price": 150, "volatility": 2},  # Goes up and down
            "SQUID_INK": {"last_price": 200, "volatility": 4}  # Large swings with pattern
        }
        
    def is_running(self):
        return self.running and self.iteration < self.max_iterations
    
    def get_current_state(self):
        # Create a simple market state for testing
        listings = {
            "RAINFOREST_RESIN": Listing("RAINFOREST_RESIN", "RAINFOREST_RESIN", "USD"),
            "KELP": Listing("KELP", "KELP", "USD"),
            "SQUID_INK": Listing("SQUID_INK", "SQUID_INK", "USD")
        }
        
        order_depths = {}
        own_trades = {}
        market_trades = {}
        position = {}
        
        # Generate dynamic order books for each product
        for product, price_data in self.price_history.items():
            # Create some sample order depth with dynamic prices
            order_depth = OrderDepth()
            
            # Simulate some price movement
            import random
            price_change = random.uniform(-price_data["volatility"], price_data["volatility"])
            current_price = price_data["last_price"] + price_change
            self.price_history[product]["last_price"] = current_price
            
            # Create buy orders slightly below current price
            order_depth.buy_orders = {
                int(current_price - 2): 10,
                int(current_price - 3): 5
            }
            
            # Create sell orders slightly above current price
            order_depth.sell_orders = {
                int(current_price + 2): -10,
                int(current_price + 3): -5
            }
            
            order_depths[product] = order_depth
            
            # Initialize empty trade history
            own_trades[product] = []
            market_trades[product] = []
            
            # Initialize position
            position[product] = 0
        
        # Create some sample market trades
        if self.iteration > 0:  # Add market trades after first iteration
            for product in self.price_history:
                price = self.price_history[product]["last_price"]
                market_trades[product] = [
                    Trade(product, int(price), 5, "buyer1", "seller1", self.current_timestamp - 50),
                    Trade(product, int(price + 1), -3, "buyer2", "seller2", self.current_timestamp - 30)
                ]
        
        # Empty observations for now
        observations = Observation({}, {})
        
        # Create the trading state
        state = TradingState(
            traderData="",
            timestamp=self.current_timestamp,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=position,
            observations=observations
        )
        
        return state
    
    def submit_orders(self, orders):
        print(f"\nIteration {self.iteration + 1} - Orders submitted:")
        for product, product_orders in orders.items():
            for order in product_orders:
                # Check position limits before executing
                new_position = self.get_new_position(product, order.quantity)
                if abs(new_position) > self.position_limits[product]:
                    print(f"- {product}: Order REJECTED - Would exceed position limit of {self.position_limits[product]}")
                    continue
                    
                print(f"- {product}: {'BUY' if order.quantity > 0 else 'SELL'} {abs(order.quantity)} @ {order.price}")
                
                # Update position based on orders (simplified matching)
                current_price = self.price_history[product]["last_price"]
                if (order.quantity > 0 and order.price >= current_price) or \
                   (order.quantity < 0 and order.price <= current_price):
                    print(f"  Order executed at {current_price}")
    
    def get_new_position(self, product, order_quantity):
        # Get current position
        current_position = 0  # In a real implementation, you'd track this
        return current_position + order_quantity
    
    def process_conversions(self, conversions):
        if conversions:
            print(f"Conversion request: {conversions}")
    
    def update_trader_data(self, trader_data):
        print(f"Updated trader data: {trader_data}")
    
    def next_iteration(self):
        self.iteration += 1
        self.current_timestamp += 100  # Increment timestamp by 100ms
        if self.iteration >= self.max_iterations:
            self.running = False

def main():
    # Create the simulation environment
    sim_env = SimpleSimulationEnvironment()
    
    # Create your trader instance
    trader = Trader()
    
    print("Starting simulation...")
    print("====================")
    
    # Run the simulation loop
    while sim_env.is_running():
        # Get the current state
        state = sim_env.get_current_state()
        
        # Run the trader's logic
        result, conversions, trader_data = trader.run(state)
        
        # Process the results
        sim_env.submit_orders(result)
        sim_env.process_conversions(conversions)
        sim_env.update_trader_data(trader_data)
        
        # Move to next iteration
        sim_env.next_iteration()
    
    print("\nSimulation completed!")
    print("====================")

if __name__ == "__main__":
    main() 