# This file contains the source code for training, validating, testing,
# and saving the model. The model is imported from "modules.py" and the 
# data loader is imported from "dataset.py".
# Losses and metrics will be plotted during training.

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from modules import VQVAE_Model
from dataset import training_loader, test_loader, validation_loader

# set device to allow GPU computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

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
    
    # len(progress_bar) = num training images / batch size
    for batch_id, inputs in progress_bar:
        # each size of input is torch.Size([32, 256, 128]), where:
        #   32 = batch size
        #   258x128 are original dimensions of HipMRI image

        inputs = inputs.to(device)
        optimiser.zero_grad()

        # first, re-shape the input for compatibility with nn.Conv2d:
        # unsqueeze() returns a new tensor with a dimension of size one 
        # inserted at the specified position. The new tensor shares the same
        # underlying data with this tensor. 
        # In our case, we need to reshape the data inputs from [32,256,128]
        # to [32,1,256,128], because nn.Conv2d will expect a 4d tensor of
        # shape [N, C, H, W] where N=batch size, C=num channels (1 for our
        # HipMRI grayscale image data), etc.
        inputs = inputs.unsqueeze(1)

        # forward pass over the model
        loss, data_reconstructed  = VQVAE_Model(inputs)

        # calculate loss, backward propagate, then have optimiser take a step
        loss = full_VQVAE_loss(loss, data_reconstructed, inputs)
        loss.backward()
        training_loss += loss.item()
        optimiser.step()

        # Update tqdm description with current loss
        progress_bar.set_postfix({'Loss': loss.item()})

    avg_loss = training_loss / len(training_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Training loss: {training_loss}')

    # Visualize HipMRI VQVAE reconstructions every few epochs
    if (epoch + 1) % 1 == 0:
        # set the model in evaluation mode - this effectively means that
        # certain layers e.g. Dropout/Batchnorm layers etc. are turned off
        VQVAE_Model.eval()
        with torch.no_grad():  # torch.no_grad() turns off gradient computations
            test_images = next(iter(test_loader))
            test_images = test_images.to(device)

            # first, re-shape the input for compatibility with nn.Conv2d:
            test_images = test_images.unsqueeze(1)
            loss, reconstructed_images = VQVAE_Model(test_images)

            # Plot original vs reconstructed
            fig, axes = plt.subplots(2, 10, figsize=(20, 4))
            for i in range(10):  # plots column-by-column
                # Original
                axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('Original', fontsize=12)

                # Reconstructed
                axes[1, i].imshow(reconstructed_images[i].cpu().squeeze(), cmap='gray')
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Reconstructed', fontsize=12)

            plt.suptitle(f'VQVAE Reconstructions - Epoch {epoch+1}', fontsize=14)
            plt.tight_layout()
            output_file = os.path.join("/home/Student/s4532467/plots", 
                                       f'VQVAE_recon_epoch{epoch}.png')
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
