import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dims: int, seq_length: int, feature_dims: int):
        super().__init__()
        self.latent_dims = latent_dims
        self.seq_length = seq_length
        self.feature_dims = feature_dims
        
        # Calculate intermediate dimensions
        self.hidden_dims = 128
        
        # Initial dense layer to get the right sequence length
        self.fc = nn.Linear(latent_dims, seq_length * self.hidden_dims)
        
        # 1D Convolutional layers with residual connections
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1),
            nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1),
            nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1)
        ])
        
        # Layer normalization for sequence dimension
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([seq_length, self.hidden_dims]) for _ in range(3)  # Use actual seq_length
        ])
        
        # Final layer to get feature dimensions
        self.final_conv = nn.Conv1d(self.hidden_dims, feature_dims, kernel_size=1)
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Initial dense layer
        x = self.fc(z)
        # Reshape to (batch_size, hidden_dims, seq_length)
        x = x.reshape(-1, self.hidden_dims, self.seq_length)
        
        # Residual blocks
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            residual = x
            x = conv(x)
            # Apply layer norm to sequence and channel dimensions
            x = x.transpose(1, 2)  # (batch_size, seq_length, hidden_dims)
            x = norm(x)  # Layer norm expects [batch_size, seq_length, hidden_dims]
            x = x.transpose(1, 2)  # (batch_size, hidden_dims, seq_length)
            x = self.activation(x)
            x = x + residual
        
        # Final layer
        x = self.final_conv(x)
        
        # Reshape to (batch_size, seq_length, feature_dims)
        x = x.transpose(1, 2)
        return x


class Discriminator(nn.Module):
    def __init__(self, seq_length: int, feature_dims: int):
        super().__init__()
        self.seq_length = seq_length
        self.feature_dims = feature_dims
        self.hidden_dims = 128
        
        # Initial conv layer
        self.initial_conv = nn.Conv1d(feature_dims, self.hidden_dims, kernel_size=1)
        
        # 1D Convolutional layers with residual connections
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1),
            nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1),
            nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=3, padding=1)
        ])
        
        # Layer normalization for sequence dimension
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([seq_length, self.hidden_dims]) for _ in range(3)  # Use actual seq_length
        ])
        
        # Global average pooling followed by dense layer
        self.fc = nn.Linear(self.hidden_dims, 1)
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_length, feature_dims)
        # Convert to (batch_size, feature_dims, seq_length)
        x = x.transpose(1, 2)
        
        # Initial conv
        x = self.initial_conv(x)
        
        # Residual blocks
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            residual = x
            x = conv(x)
            # Apply layer norm to sequence and channel dimensions
            x = x.transpose(1, 2)  # (batch_size, seq_length, hidden_dims)
            x = norm(x)  # Layer norm expects [batch_size, seq_length, hidden_dims]
            x = x.transpose(1, 2)  # (batch_size, hidden_dims, seq_length)
            x = self.activation(x)
            x = x + residual
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Final dense layer
        x = self.fc(x)
        return x


class TimeGAN(nn.Module):
    def __init__(self, latent_dims: int, seq_length: int, feature_dims: int, 
                 lr: float = 0.0002, beta1: float = 0.5, beta2: float = 0.9):
        super().__init__()
        
        self.latent_dims = latent_dims
        self.seq_length = seq_length
        self.feature_dims = feature_dims
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, seq_length * feature_dims),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(seq_length * feature_dims, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
        
        # Initialize optimizers
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr, betas=(beta1, beta2)
        )
        
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr, betas=(beta1, beta2)
        )

    def save(self, path: str):
        """Save the model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Generate fake data
        fake_data = self.generator(z)
        # Reshape to (batch_size, seq_length, feature_dims)
        return fake_data.view(-1, self.seq_length, self.feature_dims) 