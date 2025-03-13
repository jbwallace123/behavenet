""" 
Pytorch Lightning autoencoder model.
Structure from original BehaveNet code.

"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import behavenet.fitting.losses as losses
from scipy.stats import ortho_group
from torchmetrics import MeanMetric


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

        self.fc_mu = nn.Linear(self.final_size, latent_dim)
        self.fc_logvar = nn.Linear(self.final_size, latent_dim)
    
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

class VAEConvEncoder(ConvEncoder):
   
    def __init__(self, input_channels=1, latent_dim=128, kernel_size=3, stride=2, padding=1, 
                 height=140, width=170, n_labels=1, n_background=2):
        super().__init__(input_channels, latent_dim, kernel_size, stride, padding, height, width)

        # Define dimensions
        n_latents = latent_dim
        self.n_labels = n_labels
        self.n_background = n_background
        self.n_unsupervised = n_latents - n_labels - n_background

        # Linear transformations mapping NN output to label-, non-label-, and background subspaces
        self.A = nn.Linear(n_latents, n_labels, bias=False)  # supervised latents
        self.B = nn.Linear(n_latents, self.n_unsupervised, bias=False)  # unsupervised latents
        self.C = nn.Linear(n_latents, n_background, bias=True)  # background latents
        self.D = nn.Linear(n_labels, n_labels, bias=True)  # supervised latents -> labels

        # Initialize A, B, C as orthogonal matrices
        m = ortho_group.rvs(dim=n_latents).astype('float32')
        with torch.no_grad():
            self.A.weight = nn.Parameter(torch.from_numpy(m[:n_labels, :]), requires_grad=False)
            self.B.weight = nn.Parameter(torch.from_numpy(m[n_labels + n_background:, :]), requires_grad=False)
            self.C.weight = nn.Parameter(torch.from_numpy(m[n_labels:n_labels + n_background, :]), requires_grad=False)

        # Dynamically calculate the final size after convolution
        self.final_size = self.get_conv_output_shape(input_shape=(input_channels, height, width))
        print(f"Final size after convolution: {self.final_size}")

        self.fc_mu = nn.Linear(self.final_size, latent_dim)
        self.fc_logvar = nn.Linear(self.final_size, latent_dim)

    def get_conv_output_shape(self, input_shape):
        """Calculates the output shape after convolution layers."""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.encoder(x)  
            return x.shape[1]  

    def forward(self, x):
        """Processes input data and separates latent subspaces."""

        # Get encoded representation (mu, log_var from ConvEncoder)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        # Split into subspaces using mu (mean latent representation)
        z_s = self.A(mu)  # supervised latents
        z = self.B(mu)  # unsupervised latents
        z_b = self.C(mu)  # background latents

        return z_s, z_b, z, mu, log_var
    
class VAE(pl.LightningModule):
    def __init__(self, input_channels=1, latent_dim=128, kernel_size=3, stride=2, padding=1, 
                 input_height=140, input_width=170, n_labels=2, n_background=1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Encoder and Decoder
        self.encoder = VAEConvEncoder(input_channels, latent_dim, kernel_size, stride, padding, input_height, input_width, 
                                      n_labels, n_background)
        self.decoder = ConvDecoder(latent_dim, input_channels, kernel_size, stride, padding, input_height, input_width)
        
        # Loss and metrics
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []
        self.current_train_loss = 0.0
        self.current_val_loss = 0.0
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.final_latents = []

    def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
    
    def forward(self, x, use_mean=False):
        """Encodes/Decodes, samples latent (z), and reconstructs (x_hat)."""
        z_s, z_b, z, mu, log_var = self.encoder(x)
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)

        x_hat = self.decoder(z)
        # Supervised label prediction using the supervised latents (z_s)
        y_hat = self.encoder.D(z_s)  # is D the correct transformation?

        return x_hat, y_hat, z_s, z_b, mu, log_var, z

    def loss_function(self, x_hat, x, mu, log_var):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')  
        kl_loss_value = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        elbo_loss = recon_loss + kl_loss_value
        return elbo_loss

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        z_s, z_b, z, mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        loss = self.loss_function(x_hat, x, mu, log_var)
        self.current_train_loss += loss.item()

        if batch_idx == len(self.trainer.train_dataloader) - 1:  # Last batch
            self.final_latents = {'supervised': z_s, 'background': z_b, 'unsupervised': z}

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        z_s, z_b, z, mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        loss = self.loss_function(x_hat, x, mu, log_var)
        self.current_val_loss += loss.item()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, batch_size=x.shape[0])
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = self.current_train_loss / self.trainer.num_training_batches
        self.train_losses.append(avg_train_loss)
        self.current_train_loss = 0.0

    def on_validation_epoch_end(self):
        avg_val_loss = self.current_val_loss / self.trainer.num_val_batches[0]
        self.val_losses.append(avg_val_loss)
        self.current_val_loss = 0.0

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


class LatentRNN(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim) 

    def forward(self, z):
        rnn_out, _ = self.rnn(z)
        z_next = self.fc(rnn_out)  # Predict the next latent state
        return z_next[:, -1, :]  # Get the final timestep prediction
