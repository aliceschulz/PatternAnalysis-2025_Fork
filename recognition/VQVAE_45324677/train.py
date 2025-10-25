# This file contains the source code for training, validating, testing,
# and saving the model. The model is imported from "modules.py" and the 
# data loader is imported from "dataset.py".
# Losses and metrics will be plotted during training.

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import *
from plotting import *
from modules import VQVAE_Model
from dataset import training_loader, test_loader, validation_loader

# set device to allow GPU computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

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
# Time-tracking
start = time.time()
# Loss tracking
training_losses, avg_losses = [], []
validation_losses, avg_val_losses = [], []
avg_val_ssims = [] # over epochs
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
        #   256x128 are original dimensions of HipMRI image

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
        loss, data_reconstructed, encodings  = VQVAE_Model(inputs)
        #unique_indices = torch.unique(encoding_indices)
        #print(f"Unique codebook entries used: {unique_indices.numel()} / {num_embeddings}")
        #print(f'encodings: {torch.unique(encodings, return_counts=True)}')

        # calculate loss, backward propagate, then have optimiser take a step
        loss = full_VQVAE_loss(loss, data_reconstructed, inputs)
        loss.backward()
        training_loss += loss.item() * inputs.size(0)
        optimiser.step()

        # Update tqdm description with current loss
        progress_bar.set_postfix({'Loss': loss.item()})

    # now that an epoch has passed, evaluate the losses on both the training dataset, 
    # and the validation dataset. 
    avg_loss = training_loss / len(training_loader.dataset) # divides it by the entire number of training images
    avg_losses.append(avg_loss)
    training_losses.append(training_loss)  # training loss accumulates over each batch iter

    ### Validation loop ###
    VQVAE_Model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        val_ssims = [] # over every image for this epoch
        for batch_id, validation_inputs in enumerate(validation_loader):
            validation_inputs = validation_inputs.to(device)
            validation_inputs = validation_inputs.unsqueeze(1)
            N, C, H, W = validation_inputs.shape

            # forward pass over the model
            validation_loss, data_reconstructed, _  = VQVAE_Model(validation_inputs)
            validation_loss = full_VQVAE_loss(validation_loss, data_reconstructed, validation_inputs)
            running_vloss += validation_loss.item() * validation_inputs.size(0) 
            # update running training loss, multiplty each batch by the batch size. 
            #validation shape: torch.Size([32, 256, 128])
            #validation data recon shape: torch.Size([32, 1, 256, 128])
            # SSIM checking loop
            for i in range(N):
                inputs_i = validation_inputs[i].squeeze(0).cpu().numpy()
                recon_i = data_reconstructed[i].squeeze(0).cpu().numpy()
                ssim_img = ssim(inputs_i, recon_i,
                            data_range=recon_i.max() - recon_i.min())
                #print(f'ssim_img for img {i}, batch {batch_id}: {ssim_img}')
                val_ssims.append(ssim_img)
        
    
    #note: running_vloss is cumulative over the epoch,
    # but validation_loss is specific only for a certain batch. 
    avg_val_loss = running_vloss / len(validation_loader.dataset)
    avg_val_losses.append(avg_val_loss)
    validation_losses.append(running_vloss)
    avg_val_ssims.append(np.mean(val_ssims))

    print(f"""Epoch [{epoch + 1}/{num_epochs}], 
          Average Loss: {avg_loss:.4f}, 
          Training loss: {training_loss:.4f}, 
          Average Validation loss: {avg_val_loss:.4f}, 
          Validation loss: {running_vloss:.4f},
          Average SSIM (calculated on Val dataset): {np.mean(val_ssims)}""")

    # Visualize predictions after each epoch (or every few epochs)
    if plot_metrics and (epoch) % visualise_every == 0:
        show_epoch_reconstructions(model=VQVAE_Model, test_dataset=test_loader, 
                                   epoch=epoch + 1, n=10)

# note: the 'elapsed' time also includes comp time for plots
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# need to define this function in plotting.py
if plot_metrics:
    plot_training_loss(plot_save_path, avg_losses, avg_val_losses)
    plot_val_SSIMs(plot_save_path, avg_val_ssims)
