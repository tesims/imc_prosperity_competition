import requests
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("\nTesting /health:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

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

def wait_for_task(task_id: str, timeout: int = 60):
    """Wait for a task to complete with timeout"""
    start_time = time.time()
    
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for task {task_id}")
        
        response = requests.get(f"{BASE_URL}/tasks/{task_id}")
        if response.status_code != 200:
            raise RuntimeError(f"Error getting task status: {response.json()}")
        
        task_status = response.json()
        print(f"Task status: {task_status['status']}, Progress: {task_status.get('progress', 'N/A')}")
        
        if task_status['status'] == 'completed':
            return task_status
        elif task_status['status'] == 'error':
            raise RuntimeError(f"Task failed: {task_status.get('message', 'Unknown error')}")
        
        time.sleep(1)

def test_generate_synthetic():
    """Test synthetic data generation"""
    product = "BTC-USD"
    response = requests.post(f"{BASE_URL}/generate-synthetic/{product}")
    print("\nTesting /generate-synthetic/{product}:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()['task_id']
        try:
            wait_for_task(task_id)
            return True
        except (TimeoutError, RuntimeError) as e:
            print(f"Error: {e}")
            return False
    return False

def test_train_agent():
    """Test RL agent training"""
    product = "BTC-USD"
    response = requests.post(f"{BASE_URL}/train-agent/{product}")
    print("\nTesting /train-agent/{product}:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        task_id = response.json()['task_id']
        try:
            wait_for_task(task_id)
            return True
        except (TimeoutError, RuntimeError) as e:
            print(f"Error: {e}")
            return False
    return False

def test_backtest():
    """Test backtesting"""
    product = "BTC-USD"
    response = requests.post(f"{BASE_URL}/backtest/{product}")
    print("\nTesting /backtest/{product}:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def main():
    print("Starting API tests...")
    
    # Test health check
    if not test_health():
        print("Failed health check")
        return
    
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
    
    # Test agent training
    if not test_train_agent():
        print("Failed to train agent")
        return
    
    # Test backtesting
    if not test_backtest():
        print("Failed to run backtest")
        return
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 