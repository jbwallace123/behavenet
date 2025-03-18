""""
Pytorch lightning MLP decoder for BehaveNet.
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

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    """Feedforward neural network with optional 1D CNN as first layer."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        input_size = hparams['input_size']
        output_size = hparams['output_size']
        n_hidden_layers = hparams['n_hidden_layers']
        n_hidden_units = hparams['n_hidden_units']
        activation_fn = self.get_activation(hparams['activation'])
        noise_dist = hparams['noise_dist']
        lr = hparams['learning_rate']

        layers = []

        # First layer: 1D Conv (if applicable)
        if n_hidden_layers > 0:
            conv_out_size = n_hidden_units
        else:
            conv_out_size = output_size
        
        layers.append(nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_out_size,
            kernel_size=hparams['n_lags'] * 2 + 1,
            padding=hparams['n_lags']
        ))

        if activation_fn:
            layers.append(activation_fn)

        # calculate flattened size after Conv1d
        sequence_length = 255  # CHANGE to output SIZE
        flattened_size = conv_out_size * sequence_length

        # Fully Connected Layers
        in_features = flattened_size
        for i in range(n_hidden_layers):
            out_features = output_size if i == n_hidden_layers - 1 else n_hidden_units
            layers.append(nn.Linear(in_features, out_features))
            if i < n_hidden_layers - 1 and activation_fn:
                layers.append(activation_fn)
            in_features = out_features

        self.model = nn.Sequential(*layers)

        # Precision Matrix if using 'gaussian-full'
        self.precision_sqrt = (
            nn.Linear(in_features, output_size ** 2) if noise_dist == 'gaussian-full' else None
        )

        #Store loss per epoch
        self.train_losses = []
        self.val_losses = []
        self.current_train_loss = 0.0
        self.current_val_loss = 0.0

    def forward(self, x):
        """Forward pass through the network."""
        if len(x.shape) == 2:  # Input shape is [batch_size, sequence_length]
            x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, sequence_length]
        
        # Pass through the Conv1d layer
        x = self.model[0](x)  # First layer is Conv1d
        x = x.view(x.size(0), -1)  # Remove the channel dimension after Conv1d??
        
        # Pass through the remaining layers
        x = self.model[1:](x)
        
        y = None
        if self.precision_sqrt:
            y = self.precision_sqrt(x)
            y = y.view(-1, self.hparams['output_size'], self.hparams['output_size'])
            y = torch.bmm(y, y.transpose(1, 2))  # Compute precision matrix

        return x, y

    def training_step(self, batch, batch_idx):
        """Defines a single training step."""
        x, y_true = batch
        y_pred, _ = self.forward(x)
        loss = nn.MSELoss()(y_pred, y_true)

        self.current_train_loss += loss.item()
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Defines a single validation step."""
        x, y_true = batch
        y_pred, _ = self.forward(x)
        loss = nn.MSELoss()(y_pred, y_true)

        self.current_val_loss += loss.item()
        self.log("val_loss", loss, prog_bar=True)
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
        """Configures the optimizer."""
        return optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

    @staticmethod
    def get_activation(name):
        """Retrieve activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(0.05),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "linear": None,
        }
        return activations.get(name, None)
