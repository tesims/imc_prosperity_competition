import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class TimeSeriesGAN:
    def __init__(self, latent_dim: int = 64, feature_dim: int = 10):
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.generator = Generator(latent_dim, feature_dim).to(self.device)
        self.discriminator = Discriminator(feature_dim).to(self.device)
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    def generate_samples(self, n_samples: int) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            samples = self.generator(z)
        return samples.cpu().numpy()
    
    def save(self, path: str):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'latent_dim': self.latent_dim,
            'feature_dim': self.feature_dim
        }, path)
