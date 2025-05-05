from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
import traceback
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

# Pipeline components
from src.pipeline import TradingPipeline
from src.synthetic_data.train import TimeSeriesTrainer
from train_rl_agents import train_rl_agent
from src.rl_agents.enhanced_rl_agent import EnhancedRLTradingAgent
from src.rl_agents.enhanced_env import EnhancedTradingEnv

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="L2 Pipeline API", description="REST API for L2 Orderbook Pipeline")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PipelineConfig(BaseModel):
    synthetic_days: int = 30
    sequence_length: int = 20
    gan_epochs: int = 1000
    rl_episodes: int = 300
    output_dir: str = "output"

class ProductResponse(BaseModel):
    product: str
    status: str
    message: Optional[str] = None

# Global pipeline instance
pipeline = None

def initialize_pipeline(config: PipelineConfig):
    global pipeline
    base_dir = Path(config.output_dir)
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    features_dir = base_dir / "features"
    synthetic_dir = base_dir / "synthetic"
    agents_dir = base_dir / "agents"
    eval_dir = base_dir / "evaluation"
    
    for d in [raw_dir, processed_dir, features_dir, synthetic_dir, agents_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    pipeline = TradingPipeline(
        raw_data_dir=str(raw_dir),
        processed_data_dir=str(processed_dir),
        features_dir=str(features_dir),
        synthetic_dir=str(synthetic_dir),
        models_dir=str(agents_dir),
        gan_epochs=config.gan_epochs,
        rl_episodes=config.rl_episodes,
        sequence_length=config.sequence_length
    )

@app.post("/initialize")
async def initialize(config: PipelineConfig):
    """Initialize the pipeline with configuration"""
    try:
        initialize_pipeline(config)
        return {"status": "success", "message": "Pipeline initialized successfully"}
    except Exception as e:
        logger.error(f"Error in initialize: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload raw L2 market data CSV"""
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not initialized. Call /initialize first")
    
    try:
        # Read uploaded file
        contents = await file.read()
        raw_df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Determine products
        if "product" in raw_df.columns:
            products = raw_df["product"].unique()
        elif "symbol" in raw_df.columns:
            products = raw_df["symbol"].unique()
            raw_df.rename(columns={"symbol": "product"}, inplace=True)
        else:
            raise HTTPException(status_code=400, detail="CSV must contain 'product' or 'symbol' column")
        
        # Process each product
        results = []
        for product in products:
            try:
                df_prod = raw_df[raw_df["product"] == product]
                if df_prod.empty:
                    continue
                
                # Save per-product raw CSV
                raw_file = Path(pipeline.raw_data_dir) / f"{product}.csv"
                df_prod.to_csv(raw_file, index=False)
                
                results.append(ProductResponse(
                    product=product,
                    status="success",
                    message="Data uploaded successfully"
                ))
            except Exception as e:
                logger.error(f"Error processing product {product}: {str(e)}")
                logger.error(traceback.format_exc())
                results.append(ProductResponse(
                    product=product,
                    status="error",
                    message=str(e)
                ))
        
        return results
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{product}")
async def process_data(product: str):
    """Process raw data and calculate features for a specific product"""
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        # Process raw to orderbook format
        processed_df = pipeline.process_raw_data(product)
        # Feature engineering
        features_df = pipeline.calculate_features(product, processed_df)
        
        return {
            "status": "success",
            "message": f"Processed {len(processed_df)} records for {product}",
            "features_columns": features_df.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Error processing data for {product}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-synthetic/{product}")
async def generate_synthetic_data(product: str, background_tasks: BackgroundTasks):
    """Generate synthetic data for a specific product"""
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        feat_file = Path(pipeline.features_dir) / f"{product}_features.csv"
        if not feat_file.exists():
            raise HTTPException(status_code=404, detail=f"Features file not found for {product}")
        
        features_df = pd.read_csv(feat_file)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise HTTPException(status_code=400, detail=f"No numeric features for {product}")
        
        # Convert numeric features to tensor
        features_array = features_df[numeric_cols].values
        features_tensor = torch.from_numpy(features_array).float()
        
        # Prepare sequences
        sequences = pipeline.prepare_sequences(features_tensor, pipeline.sequence_length)
        if sequences is None:
            raise HTTPException(status_code=400, detail="Failed to prepare sequences")
        
        logger.info(f"Prepared sequences with shape: {sequences.shape}")
        
        # Initialize GAN trainer
        trainer = TimeSeriesTrainer(
            feature_dims=len(numeric_cols),
            seq_length=sequences.shape[1]
        )
        
        # Train GAN and generate samples in background
        def train_and_generate():
            try:
                logger.info(f"Starting GAN training for {product}")
                logger.info(f"Training parameters: epochs={pipeline.gan_epochs}, sequence_length={sequences.shape[1]}, feature_dims={len(numeric_cols)}")
                
                # Create status file to indicate training has started
                status_file = Path(pipeline.synthetic_dir) / f"{product}_synthetic.status"
                with open(status_file, 'w') as f:
                    f.write("training")
                
                trainer.train(sequences, n_epochs=pipeline.gan_epochs)
                logger.info(f"GAN training completed for {product}")
                
                minutes_per_day = 24 * 60
                n_samples = pipeline.synthetic_days * minutes_per_day
                logger.info(f"Generating {n_samples} synthetic samples for {product}")
                synthetic_df = trainer.generate_samples(n_samples=n_samples)
                
                # Add timestamp column
                start_time = pd.to_datetime(features_df['timestamp'].max()) + pd.Timedelta(minutes=1)
                synthetic_df['timestamp'] = pd.date_range(
                    start=start_time,
                    periods=len(synthetic_df),
                    freq='1min'
                )
                
                # Save synthetic data
                synthetic_out = Path(pipeline.synthetic_dir) / f"{product}_synthetic.csv"
                synthetic_df.to_csv(synthetic_out, index=False)
                logger.info(f"Saved synthetic data to {synthetic_out}")
                
                # Update status file to indicate completion
                with open(status_file, 'w') as f:
                    f.write("completed")
                
            except Exception as e:
                logger.error(f"Error in synthetic generation for {product}: {str(e)}")
                logger.error(traceback.format_exc())
                # Write error to status file
                status_file = Path(pipeline.synthetic_dir) / f"{product}_synthetic.status"
                with open(status_file, 'w') as f:
                    f.write(f"error: {str(e)}")
        
        background_tasks.add_task(train_and_generate)
        
        return {
            "status": "success",
            "message": f"Synthetic data generation started for {product}"
        }
    except Exception as e:
        logger.error(f"Error in generate_synthetic: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-agent/{product}")
async def train_agent(product: str, background_tasks: BackgroundTasks):
    """Train RL agent for a specific product"""
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        feat_file = Path(pipeline.features_dir) / f"{product}_features.csv"
        synth_file = Path(pipeline.synthetic_dir) / f"{product}_synthetic.csv"
        
        if not feat_file.exists() or not synth_file.exists():
            raise HTTPException(status_code=404, detail="Required files not found")
        
        features_df = pd.read_csv(feat_file)
        synthetic_df = pd.read_csv(synth_file)
        
        def train():
            try:
                logger.info(f"Starting RL agent training for {product}")
                agent, results = train_rl_agent(
                    product,
                    features_df,
                    synthetic_df,
                    episodes=pipeline.rl_episodes
                )
                # Save agent
                agent_path = Path(pipeline.models_dir) / f"{product}_agent.pt"
                agent.save(str(agent_path))
                logger.info(f"Saved trained agent to {agent_path}")
            except Exception as e:
                logger.error(f"Error in agent training for {product}: {str(e)}")
                logger.error(traceback.format_exc())
        
        background_tasks.add_task(train)
        
        return {
            "status": "success",
            "message": f"RL agent training started for {product}"
        }
    except Exception as e:
        logger.error(f"Error in train_agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest/{product}")
async def run_backtest(product: str):
    """Run backtesting evaluation for a specific product"""
    if not pipeline:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        feat_file = Path(pipeline.features_dir) / f"{product}_features.csv"
        agent_file = Path(pipeline.models_dir) / f"{product}_agent.pt"
        
        if not feat_file.exists() or not agent_file.exists():
            raise HTTPException(status_code=404, detail="Required files not found")
        
        features_df = pd.read_csv(feat_file)
        
        # Initialize env
        env = EnhancedTradingEnv(
            df=features_df,
            position_limit=100,
            transaction_cost=0.001,
            history_window=pipeline.sequence_length
        )
        
        # Load agent
        agent = EnhancedRLTradingAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n
        )
        agent.load(str(agent_file))
        
        # Run backtest
        obs = env.reset()
        done = False
        total_pnl = 0.0
        trades = []
        
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_pnl += reward
            if 'trade' in info:
                trades.append(info['trade'])
        
        # Save performance
        perf_file = Path(pipeline.output_dir) / "evaluation" / f"{product}_performance.csv"
        pd.DataFrame([{"total_pnl": total_pnl}]).to_csv(perf_file, index=False)
        
        return {
            "status": "success",
            "total_pnl": total_pnl,
            "num_trades": len(trades),
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 