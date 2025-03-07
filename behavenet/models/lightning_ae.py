""" 
Pytorch Lightning autoencoder model.
Structure from original BehaveNet code.

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl


class ConvEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128, kernel_size=3, stride=2, padding=1, height=140, width=170):
        super().__init__()
        # Use passed height and width instead of hardcoded values
        self.h_out = ((height + 2*padding - kernel_size) // stride + 1)  # First conv
        self.h_out = ((self.h_out + 2*padding - kernel_size) // stride + 1)  # Second conv
        self.h_out = ((self.h_out + 2*padding - kernel_size) // stride + 1)  # Third conv
        
        self.w_out = ((width + 2*padding - kernel_size) // stride + 1)  # First conv
        self.w_out = ((self.w_out + 2*padding - kernel_size) // stride + 1)  # Second conv
        self.w_out = ((self.w_out + 2*padding - kernel_size) // stride + 1)  # Third conv
        
        self.final_size = self.h_out * self.w_out * 128
        
        # Store dimensions for decoder
        self.height = height
        self.width = width
        
        self.final_size = self.h_out * self.w_out * 128

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.final_size, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=1, kernel_size=3, stride=2, padding=1, height=140, width=170):
        super().__init__()
        self.target_height = height
        self.target_width = width
        
        # Calculate intermediate dimensions
        self.h_out = height // (stride ** 3)  # For 3 transpose conv layers
        self.w_out = width // (stride ** 3)
        self.final_size = self.h_out * self.w_out * 128
        
        self.fc = nn.Linear(latent_dim, self.final_size)
        
        # Calculate output padding dynamically
        def get_output_padding(in_size, stride):
            return (in_size * stride - in_size - stride + 2)
        
        # Decoder layers with correct output padding
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, 
                             padding=padding, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride,
                             padding=padding, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.shape[0], 128, self.h_out, self.w_out)
        out = self.decoder(z)
        
        # Add size verification
        if out.shape[-2:] != (self.target_height, self.target_width):
            out = F.interpolate(out, size=(self.target_height, self.target_width), 
                              mode='bilinear', align_corners=False)
        return out

# Pytorch Lightning model
class LightningAutoencoder(pl.LightningModule):
    def __init__(self, input_channels=1, input_height=140, input_width=170, latent_dim=128, learning_rate=1e-3):
        super().__init__()
        self.encoder = ConvEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
            height=input_height,
            width=input_width
        )
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            output_channels=input_channels,
            height=input_height,
            width=input_width
        )
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

        #Store loss per epoch
        self.train_losses = []
        self.val_losses = []
        self.current_train_loss = 0.0
        self.current_val_loss = 0.0


    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)

        self.current_train_loss += loss.item()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        # Mirror the same batch handling as training_step
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        
        self.current_val_loss += loss.item()
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, batch_size=x.shape[0])
        return loss
    
    #PyTorch Lightning resets class attributes after each epoch
    #So we need to store the loss values per epoch
    def on_train_epoch_end(self):
        # Store mean epoch loss
        self.train_losses.append(self.current_train_loss / self.trainer.num_training_batches)
        self.current_train_loss = 0.0  

    def on_validation_epoch_end(self):
        self.val_losses.append(self.current_val_loss / self.trainer.num_val_batches[0])
        self.current_val_loss = 0.0  

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

