# This file contains the source code for training, validating, testing,
# and saving the model. The model is imported from "modules.py" and the 
# data loader is imported from "dataset.py".
# Losses and metrics will be plotted during training.

######################
##### Libraries ######
######################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import *
from plotting import *
from modules import VQVAE_Model, PixelCNN_Model
from dataset import training_loader, test_loader, validation_loader

#########################################
##### Device, for GPU computations ######
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

############################
##### VQVAE Optimiser ######
############################
vqvae_optimiser = optim.Adam(VQVAE_Model.parameters(), lr=learning_rate)

#######################################
##### Define VQVAE Loss function ######
#######################################
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

################################
##### VQVAE Training Loop ######
################################
start = time.time()
# Loss, SSIM tracking:
training_losses, avg_losses = [], []
validation_losses, avg_val_losses = [], []
avg_val_ssims = []
for epoch in range(num_epochs): 
    VQVAE_Model.train()  # Sets the model into training mode.
    training_loss = 0.0  # initialise training loss

    # Use tqdm to print a dynamically updating progress bar
    progress_bar = tqdm(enumerate(training_loader), 
                        total=len(training_loader), 
                        desc=f'Epoch {epoch+1}/{num_epochs}')
    
    # len(progress_bar) = num training images / batch size
    for batch_id, inputs in progress_bar:
        # each size of input is torch.Size([N, 256, 128]), where:
        #   N = batch size
        #   256x128 are original dimensions of HipMRI image

        inputs = inputs.to(device)
        vqvae_optimiser.zero_grad()

        # For compatibility with nn.Conv2d, reshape from [N,H,W]->[N,C,H,W]
        inputs = inputs.unsqueeze(1)

        # forward pass over the model
        loss, data_reconstructed, encodings  = VQVAE_Model(inputs)
    
        # calculate loss, backward propagate, then have optimiser take a step
        loss = full_VQVAE_loss(loss, data_reconstructed, inputs)
        loss.backward()
        training_loss += loss.item() * inputs.size(0)
        vqvae_optimiser.step()

        # Update tqdm description with current loss
        progress_bar.set_postfix({'Loss': loss.item()})

    # now that an epoch has passed, evaluate the losses on both the training dataset, 
    # and the validation dataset. 
    avg_loss = training_loss / len(training_loader.dataset)
    avg_losses.append(avg_loss)
    training_losses.append(training_loss)

    ##################################
    ##### VQVAE Validation Loop ######
    ##################################
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
        
            # SSIM checking loop:
            for i in range(N):
                inputs_i = validation_inputs[i].squeeze(0).cpu().numpy()
                recon_i = data_reconstructed[i].squeeze(0).cpu().numpy()
                ssim_img = ssim(inputs_i, recon_i,
                            data_range=recon_i.max() - recon_i.min())
                val_ssims.append(ssim_img)
        
    
    # note: running_vloss is cumulative over the epoch,
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

    # Visualize reconstructions after 'visualise_every' number of epochs
    if plot_metrics and (epoch) % visualise_every == 0:
        show_epoch_reconstructions(model=VQVAE_Model, test_dataset=test_loader, 
                                   epoch=epoch + 1, n=10)

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

if plot_metrics:
    plot_training_loss(plot_save_path, avg_losses, avg_val_losses)
    plot_val_SSIMs(plot_save_path, avg_val_ssims)


###################################
##### PixelCNN Training Loop ######
###################################
print("Now training the PixelCNN...")
start = time.time()
pixel_training_losses = []
pixelcnn_optimiser = optim.Adam(PixelCNN_Model.parameters(), lr=learning_rate)
VQVAE_Model.to(device) # taking already trained VQVAE
for epoch in range(num_epochs_pixelCNN):
    total_loss = 0.0
    PixelCNN_Model.train()
    progress_bar = tqdm(enumerate(training_loader), 
                        total=len(training_loader), 
                        desc=f'Epoch {epoch+1}/{num_epochs_pixelCNN}')
    
    for batch_id, inputs in progress_bar:
        inputs = inputs.to(device)
        with torch.no_grad():
            inputs = inputs.unsqueeze(1)
            latents = VQVAE_Model.encode(inputs).to(device)
            latents_embedded = VQVAE_Model._VQ._embedding(latents).permute(0,3,1,2).float()
            latents_embedded = latents_embedded.to(device)
                
        # latents shape: torch.Size([32, 64, 32])
        logits = PixelCNN_Model(latents_embedded)
        # logits shape: torch.Size([32, 256, 64, 32])
        loss = F.cross_entropy(logits, latents)

        pixelcnn_optimiser.zero_grad()
        loss.backward()
        pixelcnn_optimiser.step()
        total_loss += loss.item()
    train_loss = total_loss / len(training_loader)
    pixel_training_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs_pixelCNN}] Loss: {train_loss:.4f}")
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

######################################################
##### Use PixelCNN and VQVAE to generate images ######
######################################################
def generate_images(shape, num_embeddings):
    """Generate images, using the trained PixelCNN_Model.
    
    Utilises the Decoder module to return a constructed image,
    based on an input discrete code.

    Args:
        shape (tuple): tuple of (N, H, W) where N is the batch size,
            H and W are the image dimensions.

    PreReqs:
        Assumed PixelCNN_Model (& VQVAE) is already trained
        N, H, W are integers > 0 
    """
    PixelCNN_Model.eval()
    VQVAE_Model.eval()
    N, H, W, = shape
    # generates a 'blank' image: 
    latents = torch.zeros((N, H, W), dtype=torch.long, device=device)
    
    latents_embedded = VQVAE_Model._VQ._embedding(latents).permute(0,3,1,2).float()
    latents_embedded = latents_embedded.to(device)
    for i in range(H):
        for j in range(W):
            with torch.no_grad():
                # predict the next code/pixel
                logits = PixelCNN_Model(latents_embedded)

                # sampling from predicted distribution, by using softmax.
                probs = F.softmax(logits[:, :, i, j], dim=-1)
                latents[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)
                #latents shape: torch.Size([4, 256, 128])
            
    embedding_weights = VQVAE_Model._VQ._embedding.weight # (num_embeddings x embedding_dim)
    quantised = embedding_weights[latents]
    quantised = quantised.permute(0,3,1,2).contiguous()   
    generated_img = VQVAE_Model.decode(quantised)
    # latents.shape [4,64,32]
    # embedding_weights.shape [256,32]
    # quantised.shape [4,32,64,32]
    # generated_img.shape [4,1,256,128]

    return generated_img

generated_images = generate_images(shape=(4, 64, 32), num_embeddings=VQVAE_Model.get_embeddings())

if plot_metrics:
    plot_PixelCNN_loss(plot_save_path, pixel_training_losses, val_losses=None)
    plot_generated_images(generated_images, num_epochs_pixelCNN)