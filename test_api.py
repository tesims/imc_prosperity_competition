import requests
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_initialize():
    """Test pipeline initialization"""
    config = {
        "synthetic_days": 1,  # Reduced for testing
        "sequence_length": 10,  # Reduced for testing
        "gan_epochs": 10,  # Reduced for testing
        "rl_episodes": 5,  # Reduced for testing
        "output_dir": "output"
    }
    response = requests.post(f"{BASE_URL}/initialize", json=config)
    print("\nTesting /initialize:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_upload():
    """Test data upload with sample data"""
    # Create sample L2 orderbook data
    n_records = 20  # Reduced for testing
    timestamps = pd.date_range(start='2024-01-01', periods=n_records, freq='1min')
    data = []
    
    for ts in timestamps:
        # Generate 5 levels of bids and asks
        base_price = 50000
        for i in range(5):
            # Add bid
            data.append({
                'timestamp': ts,
                'product': 'BTC-USD',
                'side': 'BID',
                'price': base_price - (i * 10),  # Decreasing prices for bids
                'quantity': np.random.uniform(0.1, 1.0),
                'order_id': f'bid_{ts}_{i}',
                'type': 'limit'
            })
            # Add ask
            data.append({
                'timestamp': ts,
                'product': 'BTC-USD',
                'side': 'ASK',
                'price': base_price + ((i + 1) * 10),  # Increasing prices for asks
                'quantity': np.random.uniform(0.1, 1.0),
                'order_id': f'ask_{ts}_{i}',
                'type': 'limit'
            })
    
    df = pd.DataFrame(data)
    
    # Save to temporary CSV
    temp_file = "temp_test_data.csv"
    df.to_csv(temp_file, index=False)
    
    # Upload file
    with open(temp_file, 'rb') as f:
        files = {'file': ('test_data.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    # Clean up
    Path(temp_file).unlink()
    
    print("\nTesting /upload:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_process():
    """Test data processing"""
    product = "BTC-USD"
    response = requests.post(f"{BASE_URL}/process/{product}")
    print("\nTesting /process/{product}:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_generate_synthetic():
    """Test synthetic data generation"""
    product = "BTC-USD"
    response = requests.post(f"{BASE_URL}/generate-synthetic/{product}")
    print("\nTesting /generate-synthetic/{product}:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_train_agent():
    """Test RL agent training"""
    product = "BTC-USD"
    response = requests.post(f"{BASE_URL}/train-agent/{product}")
    print("\nTesting /train-agent/{product}:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_backtest():
    """Test backtesting"""
    product = "BTC-USD"
    response = requests.post(f"{BASE_URL}/backtest/{product}")
    print("\nTesting /backtest/{product}:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def wait_for_file(filepath: str, timeout: int = 60):
    """Wait for a file to exist with timeout"""
    start_time = time.time()
    status_file = str(Path(filepath).parent / f"{Path(filepath).stem}.status")
    
    while not Path(filepath).exists():
        if time.time() - start_time > timeout:
            # Check status file for error
            if Path(status_file).exists():
                with open(status_file, 'r') as f:
                    status = f.read().strip()
                    if status.startswith('error:'):
                        raise RuntimeError(f"Generation failed: {status}")
            raise TimeoutError(f"Timeout waiting for {filepath}")
        
        # Check status file
        if Path(status_file).exists():
            with open(status_file, 'r') as f:
                status = f.read().strip()
                if status.startswith('error:'):
                    raise RuntimeError(f"Generation failed: {status}")
                print(f"Current status: {status}")
        
        time.sleep(1)
    print(f"File {filepath} is ready")

def main():
    print("Starting API tests...")
    
    # Test initialization
    if not test_initialize():
        print("Failed to initialize pipeline")
        return
    
    # Test upload
    if not test_upload():
        print("Failed to upload data")
        return
    
    # Test processing
    if not test_process():
        print("Failed to process data")
        return
    
    # Test synthetic data generation
    if not test_generate_synthetic():
        print("Failed to generate synthetic data")
        return
    
    # Wait for synthetic data generation to complete
    print("\nWaiting for synthetic data generation to complete...")
    try:
        wait_for_file("output/synthetic/BTC-USD_synthetic.csv")
    except TimeoutError as e:
        print(f"Error: {e}")
        return
    
    # Test agent training
    if not test_train_agent():
        print("Failed to train agent")
        return
    
    # Wait for agent training to complete
    print("\nWaiting for agent training to complete...")
    try:
        wait_for_file("output/agents/BTC-USD_agent.pt")
    except TimeoutError as e:
        print(f"Error: {e}")
        return
    
    # Test backtesting
    if not test_backtest():
        print("Failed to run backtest")
        return
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 