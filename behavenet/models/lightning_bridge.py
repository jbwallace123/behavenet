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

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class NeuralToVideoBridge(pl.LightningModule):
    """
    Bridge module that connects a neural network (MLP) to a VAE for video reconstruction.
    requires a pre-trained MLP and VAE model.
    """
    def __init__(self, mlp, vae, latent_dim, learning_rate=1e-3):
        super().__init__()
        self.mlp = mlp
        self.vae = vae
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        # Freeze the VAE encoder/decoder
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # Linear mapping from MLP output to VAE latent space (μ)
        self.linear_map = nn.Linear(mlp.hparams['output_size'], latent_dim)

    def forward(self, neural_input):
        mlp_latent, _ = self.mlp(neural_input)
        z_hat = self.linear_map(mlp_latent)  # Predict μ
        recon_video = self.vae.decoder(z_hat)  # Decode via VAE
        return recon_video, z_hat

    def training_step(self, batch, batch_idx):
        neural_input, video_frame = batch
        recon_video, z_hat = self.forward(neural_input)

        # Get true latent mean μ from VAE encoder
        with torch.no_grad():
            _, _, _, mu, _, _ = self.vae.encoder(video_frame)

        # Latent loss: encourage z_hat ~ μ
        latent_loss = F.mse_loss(z_hat, mu)

        # Optional recon loss if you want to optimize output frame too
        with torch.no_grad():
            target_frame = video_frame  # Or apply transformations if needed
        recon_loss = F.mse_loss(recon_video, target_frame)

        total_loss = latent_loss + recon_loss
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return optim.Adam(self.linear_map.parameters(), lr=self.learning_rate)