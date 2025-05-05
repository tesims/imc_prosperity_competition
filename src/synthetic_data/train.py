import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from .models import TimeGAN
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

class TimeSeriesTrainer:
    def __init__(
        self,
        feature_dims: int,
        seq_length: int,
        latent_dims: int = 100,
        gradient_penalty_weight: float = 10.0,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.9
    ):
        """
        Initialize TimeSeriesTrainer for synthetic data generation.
        
        Args:
            feature_dims: Number of features in the data
            seq_length: Length of the time series sequence
            latent_dims: Dimension of the latent space
            gradient_penalty_weight: Weight of gradient penalty term
            lr: Learning rate for optimizers
            beta1: Beta1 parameter for Adam optimizer
            beta2: Beta2 parameter for Adam optimizer
        """
        self.feature_dims = feature_dims
        self.seq_length = seq_length
        self.latent_dims = latent_dims
        self.gradient_penalty_weight = gradient_penalty_weight
        
        # Initialize GAN with improved architecture
        self.gan = TimeGAN(
            latent_dims=latent_dims,
            seq_length=seq_length,
            feature_dims=feature_dims,
            lr=lr,
            beta1=beta1,
            beta2=beta2
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gan = self.gan.to(self.device)
        
        # Initialize feature scaling
        self.feature_scaler: Optional[Dict[str, Tuple[float, float]]] = None
        
        # Create checkpoint directory
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _scale_features(self, data: torch.Tensor) -> torch.Tensor:
        """Scale features to [-1, 1] range."""
        # Data is already a tensor with shape (n_sequences, seq_length, n_features)
        
        # Initialize feature_scaler if it doesn't exist
        if self.feature_scaler is None:
            self.feature_scaler = {}
        
        # Get min and max values for each feature
        min_vals = data.min(dim=0)[0].min(dim=0)[0]  # Shape: (n_features,)
        max_vals = data.max(dim=0)[0].max(dim=0)[0]  # Shape: (n_features,)
        
        # Store scaling factors
        for i in range(self.feature_dims):
            self.feature_scaler[f'feature_{i}'] = (min_vals[i].item(), max_vals[i].item())
        
        # Check for features with zero range
        zero_range_features = (max_vals - min_vals) < 1e-8
        if zero_range_features.any():
            logger.warning(f"Found {zero_range_features.sum().item()} features with near-zero range. Adding small noise.")
            # Add tiny noise to constant features to avoid division by zero
            noise_mask = torch.zeros_like(data)
            for i in range(self.feature_dims):
                if (max_vals[i] - min_vals[i]) < 1e-8:
                    noise = torch.randn(data.shape[0], data.shape[1], 1, device=data.device) * 1e-4
                    noise_mask[:, :, i:i+1] = noise
            data = data + noise_mask
            
            # Recalculate min and max
            min_vals = data.min(dim=0)[0].min(dim=0)[0]
            max_vals = data.max(dim=0)[0].max(dim=0)[0]
            
            # Update scaler
            for i in range(self.feature_dims):
                self.feature_scaler[f'feature_{i}'] = (min_vals[i].item(), max_vals[i].item())
        
        # Scale data - using epsilon to avoid division by zero
        eps = 1e-8
        scaled_data = 2 * (data - min_vals) / (max_vals - min_vals + eps) - 1
        
        # Clamp values to ensure they're in [-1, 1]
        scaled_data = torch.clamp(scaled_data, -1.0, 1.0)
        
        return scaled_data
    
    def train_step(self, real_data: torch.Tensor) -> Tuple[float, float]:
        """Perform one training step"""
        batch_size = real_data.size(0)
        
        # Train discriminator
        self.gan.discriminator_optimizer.zero_grad()
        
        # Real data
        d_real = self.gan.discriminator(real_data.view(batch_size, -1))
        d_real_loss = -torch.mean(d_real)
        
        # Fake data
        z = torch.randn(batch_size, self.latent_dims, device=self.device)
        fake_data = self.gan.generator(z)
        d_fake = self.gan.discriminator(fake_data.detach().view(batch_size, -1))
        d_fake_loss = torch.mean(d_fake)
        
        # Gradient penalty
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand(batch_size, self.seq_length * self.feature_dims)
        interpolates = alpha * real_data.view(batch_size, -1) + (1 - alpha) * fake_data.detach().view(batch_size, -1)
        interpolates.requires_grad_(True)
        d_interpolates = self.gan.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=self.device),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_penalty = self.gradient_penalty_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss + gradient_penalty
        d_loss.backward()
        self.gan.discriminator_optimizer.step()
        
        # Train generator
        self.gan.generator_optimizer.zero_grad()
        
        fake_data = self.gan.generator(z)
        g_fake = self.gan.discriminator(fake_data.view(batch_size, -1))
        g_loss = -torch.mean(g_fake)
        
        g_loss.backward()
        self.gan.generator_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def train(self, real_data: torch.Tensor, n_epochs: int = 100, batch_size: int = 64, n_critic: int = 5,
              gradient_penalty_weight: float = 10.0, checkpoint_dir: str = None) -> Dict[str, List[float]]:
        """
        Train the GAN model using Wasserstein loss with gradient penalty.
        
        Args:
            real_data: Training data tensor of shape (n_sequences, sequence_length, n_features)
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            n_critic: Number of discriminator updates per generator update
            gradient_penalty_weight: Weight of gradient penalty term
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary containing training metrics
        """
        # Ensure data is on correct device and has contiguous memory layout
        real_data = real_data.to(self.device).contiguous()
        
        # Validate input dimensions
        if len(real_data.shape) != 3:
            raise ValueError(f"Expected real_data shape (n_sequences, sequence_length, n_features), got {real_data.shape}")
        
        n_sequences, sequence_length, n_features = real_data.shape
        if sequence_length != self.seq_length:
            raise ValueError(f"Sequence length mismatch. Expected {self.seq_length}, got {sequence_length}")
        if n_features != self.feature_dims:
            raise ValueError(f"Feature dimensions mismatch. Expected {self.feature_dims}, got {n_features}")
        
        # Scale features to [-1, 1] range
        scaled_data = self._scale_features(real_data)
        
        # Create data loader
        dataset = TensorDataset(scaled_data)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Training metrics
        g_losses = []
        d_losses = []
        
        for epoch in range(n_epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for batch_idx, (real_batch,) in enumerate(data_loader):
                current_batch_size = real_batch.size(0)
                
                # Train discriminator
                for _ in range(n_critic):
                    d_loss, g_loss = self.train_step(real_batch)
                    epoch_d_losses.append(d_loss)
                    epoch_g_losses.append(g_loss)
                
                # Record average losses for the epoch
                g_losses.append(np.mean(epoch_g_losses))
                d_losses.append(np.mean(epoch_d_losses))
            
            # Print progress and save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] "
                      f"D_loss: {d_losses[-1]:.4f} "
                      f"G_loss: {g_losses[-1]:.4f}")
                
                if checkpoint_dir:
                    self.save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        return {
            "generator_losses": g_losses,
            "discriminator_losses": d_losses
        }
    
    def generate_samples(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples and rescale them to original range."""
        if self.gan is None or self.feature_scaler is None:
            raise ValueError("Must train GAN before generating samples")
        
        self.gan.generator.eval()
        with torch.no_grad():
            # Generate samples in batches to avoid memory issues
            batch_size = 32
            num_batches = (n_samples + batch_size - 1) // batch_size
            all_samples = []
            
            for i in range(num_batches):
                current_batch_size = min(batch_size, n_samples - i * batch_size)
                # Generate latent vectors
                z = torch.randn(current_batch_size, self.latent_dims, device=self.device)
                # Generate data from latent vectors
                generated = self.gan.generator(z)
                # Ensure proper reshaping
                batch_samples = generated.reshape(current_batch_size, self.seq_length, self.feature_dims).contiguous()
                # Move to CPU and convert to numpy
                batch_samples = batch_samples.cpu().numpy()
                all_samples.append(batch_samples)
            
            # Combine all batches
            samples = np.vstack([batch.reshape(-1, self.feature_dims) for batch in all_samples])
            # Trim to exact number of samples needed
            samples = samples[:n_samples]
            
            # Create DataFrame
            feature_names = [f'feature_{i}' for i in range(self.feature_dims)]
            df = pd.DataFrame(samples, columns=feature_names)
            
            # Rescale samples
            for i, col in enumerate(df.columns):
                scaler_key = f'feature_{i}'
                if scaler_key in self.feature_scaler:
                    min_val, max_val = self.feature_scaler[scaler_key]
                    if abs(max_val - min_val) > 1e-8:  # Avoid division by zero
                        df[col] = (df[col] + 1) / 2 * (max_val - min_val) + min_val
        
        return df
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        self.gan.save(str(path))
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")
        self.gan.load(str(path))
