# This file contains the source code for training, validating, testing,
# and saving the model. The model is imported from "modules.py" and the 
# data loader is imported from "dataset.py".
# Losses and metrics will be plotted during training.

import torch.optim as optim
from modules.py import VQVAE_Model
from dataset.py import training_loader, test_loader, validation_loader
from tqdm import tqdm

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

# Training loop
for epoch in range(num_epochs): 
    VQVAE_Model.train()  # Sets the model into training mode.
    training_loss = 0.0  # initialise training loss

    # Use tqdm to create an "iterable object that acts exactly
    # like the original iterable, but prints a dynamically updating
    # progress bar every time a value is requested."
    progress_bar = tqdm(enumerate(training_loader), 
                        total=len(training_loader), 
                        desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, (inputs, _) in progress_bar:

        inputs = inputs.to(device)
        # calculate loss, backward propagate, then have optimiser take a step
        optimiser.zero_grad()
        loss, data_reconstructed  = VQVAE_Model(inputs)
        loss = full_VQVAE_loss(loss, data_reconstructed, inputs)
        loss.backward()
        training_loss += loss.item()
        optimiser.step()

        # Update tqdm description with current loss
        progress_bar.set_postfix({'Loss': loss.item()})

    avg_loss = training_loss / len(training_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], 
          Average Loss: {avg_loss:.4f}, 
          Training loss: {training_loss}')