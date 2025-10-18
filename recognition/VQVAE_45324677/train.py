# This file contains the source code for training, validating, testing,
# and saving the model. The model is imported from "modules.py" and the 
# data loader is imported from "dataset.py".
# Losses and metrics will be plotted during training.

import torch.optim as optim
from modules.py import VQVAE_Model
from dataset.py import training_loader, test_loader, validation_loader

# Hyperparameters
num_epochs = 2
learning_rate = 1e-4

# Optimiser
optimiser = optim.Adam(VQVAE_Model.parameters(), lr=learning_rate)

# Fully-defined loss function
# note that VQVAE_Model.forward(inputs) returns loss and the x_reconstructed,
# but the full loss function for VQVAE needs to include reconstruction loss. 
def full_VQVAE_loss(VQ_loss, x_reconstructed, inputs):
    """Compute and return the full VQVAE loss.
    
    i.e, reconstruction loss + codebook loss + commitment loss. Compute
    the reconstruction loss as the MSE between the original input and 
    the reconstructed image, to help optimise the encoder and decoder. 
    
    Args:
        VQ_loss (float): Loss obtained from the VectorQuantiser class.
            Consists of codebook loss + commitment loss.
        x_reconstructed (np.ndarray): Reconstructed HipMRI image, i.e., 
            after encoder representation and decoder reconstruction.
        inputs: original HipMRI images (as represented in array format)
    """

    reconstruction_loss = F.mse_loss(x_reconstructed, inputs) 

    VQVAE_loss = reconstruction_loss + VQ_loss
    return VQVAE_loss



