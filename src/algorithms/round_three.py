from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List
import jsonpickle
import math



class Trader:
    def __init__(self):
        # Initialize empty dictionaries for all products
        self.price_history = {}
        self.price_stats = {}
        self.POSITION_LIMITS = {}
        
        # Default parameters for any product
        self.DEFAULT_PARAMS = {
            "zscore_threshold": 0.5,
            "momentum_weight": 0.6,
            "min_position": 15,
            "volatility_threshold": 1.2
        }
        
        # Window sizes for calculations
        self.HISTORY_WINDOW = 30    # Keep longer window for better trend detection
        self.SHORT_WINDOW = 5       # Quick reaction to price changes
        self.LONG_WINDOW = 15       # Trend confirmation
        
        # Trading parameters - More aggressive
        self.MIN_TRADES = 2         # Reduced minimum trades
        self.TREND_THRESHOLD = 0.2  # More sensitive trend detection
        self.BASE_POSITION_SIZE = 15  # Increased base position
        
        # Dynamic position sizing parameters
        self.MIN_POSITION_MULT = 0.7    # Higher minimum multiplier
        self.MAX_POSITION_MULT = 3.0    # More aggressive maximum multiplier
        self.MOMENTUM_SCALING = 2.0     # Stronger momentum impact
        self.VOLATILITY_SCALING = 1.0   # Reduced volatility impact
    
    # The Trader class does not require an __init__ method because the AWS Lambda environment is stateless.
    # Instead, state persistence between calls can be maintained using the traderData string.
    
    def run(self, state: TradingState):
        # First, try to load previous state if available
        if state.traderData:
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.price_history = saved_state.get("price_history", self.price_history)
                self.price_stats = saved_state.get("price_stats", self.price_stats)
            except:
                pass  # If there's any error loading state, keep using current state

        # Debugging: print out incoming state for reference.
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Initialize the result dictionary to store orders per product.
        result = {}

        # Iterate over each product that has an order depth in the state.
        for product in state.order_depths:
            # Initialize product if we haven't seen it before
            self.initialize_product(product)
            
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Get current position for the product
            current_position = state.position.get(product, 0)
            
            # Calculate acceptable price and signal strength
            acceptable_price, signal_strength = self.calculate_acceptable_price(product, state)
            
            if acceptable_price is None:
                continue
                
            # Calculate dynamic position size based on signal strength
            target_position_size = self.calculate_position_size(product, signal_strength, acceptable_price)
            
            # Calculate remaining position capacity with dynamic sizing
            buy_capacity = min(self.POSITION_LIMITS[product] - current_position, target_position_size)
            sell_capacity = min(self.POSITION_LIMITS[product] + current_position, target_position_size)

            print(f"{product} - Signal strength: {signal_strength:.2f}, Target size: {target_position_size}")
            print(f"{product} - Acceptable price: {acceptable_price}, Current price: {current_position}")

            # Create BUY orders if sell orders (asks) are below our acceptable price
            if len(order_depth.sell_orders) != 0 and buy_capacity > 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    # Use dynamic position sizing
                    quantity = min(-best_ask_amount, buy_capacity)
                    print(f"BUY {quantity}x {best_ask} (Signal: {signal_strength:.2f})")
                    orders.append(Order(product, best_ask, quantity))

            # Create SELL orders if buy orders (bids) are above our acceptable price
            if len(order_depth.buy_orders) != 0 and sell_capacity > 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    # Use dynamic position sizing
                    quantity = min(best_bid_amount, sell_capacity)
                    print(f"SELL {quantity}x {best_bid} (Signal: {signal_strength:.2f})")
                    orders.append(Order(product, best_bid, -quantity))

            # Add the list of orders for the product to the overall result.
            result[product] = orders
    
        # ---------------------------------------------------------------
        # STEP 4: Update and persist internal state using traderData.
        # You can use jsonpickle to serialise any internal variables
        # (e.g., positions, historical data, parameters) into a string.
        #
		# String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = self.serialize_state(state)

        # ---------------------------------------------------------------
        # STEP 5: Determine conversion requests if required.
        # Conversion requests can be used to transform positions according to
        # the challenge rules. Return an integer value (or 0/None if no conversion).
        conversions = self.determine_conversions(state)
        
        # Return a tuple containing:
        # - result: Dictionary mapping product symbols to a list of Order objects.
        # - conversions: Conversion request value.
        # - traderData: Updated state string for persistence in subsequent iterations.
        
		# Sample conversion request. Check more details below. 
        conversions = 0
        return result, conversions, traderData
    
    # ---------------------------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------------------------
    
    def initialize_product(self, product: str):
        """Initialize data structures for a new product"""
        if product not in self.price_history:
            self.price_history[product] = []
            self.price_stats[product] = {"mean": None, "std": None}
            self.POSITION_LIMITS[product] = 50  # Default position limit
    
    def calculate_statistics(self, product: str):
        """
        Calculate mean and standard deviation for normalization
        """
        history = self.price_history[product]
        if len(history) > 0:
            mean = sum(history) / len(history)
            squared_diff = sum((x - mean) ** 2 for x in history)
            std = math.sqrt(squared_diff / len(history)) if len(history) > 1 else 1
            self.price_stats[product] = {"mean": mean, "std": std if std > 0 else 1}
    
    def normalize_price(self, price: float, product: str) -> float:
        """
        Normalize price using z-score normalization
        """
        stats = self.price_stats[product]
        if stats["mean"] is not None and stats["std"] is not None:
            return (price - stats["mean"]) / stats["std"]
        return price
    
    def denormalize_price(self, normalized_price: float, product: str) -> float:
        """
        Convert normalized price back to original scale
        """
        stats = self.price_stats[product]
        if stats["mean"] is not None and stats["std"] is not None:
            return (normalized_price * stats["std"]) + stats["mean"]
        return normalized_price

    def calculate_moving_averages(self, history):
        """Calculate short and long moving averages"""
        if len(history) < self.LONG_WINDOW:
            return None, None
            
        short_ma = sum(history[-self.SHORT_WINDOW:]) / self.SHORT_WINDOW
        long_ma = sum(history[-self.LONG_WINDOW:]) / self.LONG_WINDOW
        return short_ma, long_ma
    
    def calculate_volatility(self, history):
        """Calculate recent price volatility"""
        if len(history) < self.SHORT_WINDOW:
            return None
        
        mean = sum(history[-self.SHORT_WINDOW:]) / self.SHORT_WINDOW
        squared_diff = [(x - mean) ** 2 for x in history[-self.SHORT_WINDOW:]]
        return (sum(squared_diff) / self.SHORT_WINDOW) ** 0.5
    
    def calculate_zscore(self, price, mean, std):
        """Calculate z-score for mean reversion"""
        if std == 0:
            return 0
        return (price - mean) / std
    
    def get_trend_strength(self, short_ma, long_ma, volatility):
        """Calculate trend strength relative to volatility"""
        if volatility == 0:
            return 0
        return (short_ma - long_ma) / volatility
    
    def calculate_position_size(self, product: str, signal_strength: float, current_price: float) -> int:
        """
        Calculate dynamic position size based on multiple factors:
        - Signal strength (momentum/mean reversion)
        - Price volatility
        - Recent trend direction
        - Current market conditions
        """
        params = self.DEFAULT_PARAMS
        history = self.price_history[product]
        
        if len(history) < self.SHORT_WINDOW:
            return params["min_position"]
        
        # Calculate trend strength
        short_ma, long_ma = self.calculate_moving_averages(history)
        trend_strength = (short_ma - long_ma) / long_ma if long_ma else 0
        
        # Calculate volatility ratio
        current_volatility = self.calculate_volatility(history[-self.SHORT_WINDOW:])
        volatility_ratio = current_volatility / params["volatility_threshold"] if current_volatility else 1
        
        # Base position size calculation
        position_mult = 1.0
        
        # Adjust for momentum
        momentum_factor = abs(signal_strength) * self.MOMENTUM_SCALING
        position_mult *= (1 + momentum_factor)
        
        # Adjust for trend alignment
        if abs(trend_strength) > self.TREND_THRESHOLD:
            trend_alignment = 1 + (abs(trend_strength) * 0.5)
            position_mult *= trend_alignment
        
        # Adjust for volatility
        volatility_factor = 1 / (volatility_ratio ** 0.5)  # Square root to dampen effect
        position_mult *= volatility_factor
        
        # Clamp multiplier
        position_mult = max(self.MIN_POSITION_MULT, min(self.MAX_POSITION_MULT, position_mult))
        
        # Calculate final position size
        base_size = max(params["min_position"], self.BASE_POSITION_SIZE)
        position_size = int(base_size * position_mult)
        
        # Ensure within limits
        return min(position_size, self.POSITION_LIMITS[product])

    def calculate_acceptable_price(self, product: str, state: TradingState) -> tuple:
        """
        Calculate acceptable price and signal strength using enhanced logic
        """
        # Initialize product if needed
        self.initialize_product(product)
        
        order_depth = state.order_depths[product]
        
        # Get current mid price
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        
        if best_bid is None or best_ask is None:
            return None, 0
            
        mid_price = (best_bid + best_ask) / 2
        
        # Update price history and statistics
        self.price_history[product].append(mid_price)
        if len(self.price_history[product]) > self.HISTORY_WINDOW:
            self.price_history[product] = self.price_history[product][-self.HISTORY_WINDOW:]
        self.calculate_statistics(product)
        
        # Calculate trend indicators
        short_ma, long_ma = self.calculate_moving_averages(self.price_history[product])
        if short_ma is None or long_ma is None:
            return mid_price, 0
            
        # Calculate momentum and mean reversion signals
        momentum_signal = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
        zscore = self.normalize_price(mid_price, product)
        
        # Combine signals based on product parameters
        params = self.DEFAULT_PARAMS
        signal_strength = (params["momentum_weight"] * momentum_signal + 
                         (1 - params["momentum_weight"]) * -zscore)  # Negative zscore for mean reversion
        
        # Calculate price adjustment
        price_adjustment = signal_strength * self.price_stats[product]["std"]
        acceptable_price = mid_price + price_adjustment
        
        return acceptable_price, signal_strength

    def serialize_state(self, state: TradingState) -> str:
        """
        Serialize the internal state including price history and statistics
        """
        internal_state = {
            "positions": state.position,
            "price_history": self.price_history,
            "price_stats": self.price_stats
        }
        return jsonpickle.encode(internal_state)
    
    def determine_conversions(self, state: TradingState) -> int:
        """
        Determine the conversion request based on the current positions and any conversion limits.
        For example, if you have an excess position that can be converted (and have the necessary
        fees covered), return a conversion request integer. Return 0 or None if no conversion is needed.
        
        In the provided sample, simply returning 0.
        """
        # Replace this with your conversion decision logic.
        return 0