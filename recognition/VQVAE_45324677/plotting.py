# A script to separate out the specific plotting functions.
# Acknowledgement: This script was adapted from Dr Wei Dai, as provided in
# the "UNet_segmentation_code_demo.ipynb" file provided for COMP3710. 

######################
##### Libraries ######
######################
import matplotlib.pyplot as plt
import torch
import os
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################
##### Plot original HipMRI Prostate Images ######
#################################################
def plt_original_imgs(image, plot_save_path, index):
    """Plot original HipMRI images, to see what data looks like.

    Args: 
        image: np.ndarray, image to be plotted. Often will be output from 
               load_data_2D function.
        save_path: str, directory to the save location
        index: int, image index. 
    """
    if image.ndim == 3:
        image = image.squeeze()

    plt.imshow(image, cmap='gray')
    plt.title('Image of HipMRI Study on Prostate Cancer')
    plt.axis('off')

    os.makedirs(plot_save_path, exist_ok=True)
    output_file = os.path.join(plot_save_path, f'HipMRI_img_{index}.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

#######################################
##### Plot VQVAE Reconstructions ######
#######################################
def show_epoch_reconstructions(model, test_dataset, epoch: int, n: int):
    """Show model reconstructions after a specific epoch.
    
    Args:
        model: of class nn.Module. 
        test_dataset: DataLoader class, containing the test images.
        epoch (int): Specified within training loop; provides an epoch
            number that can be included in plot titles. 
        n (int): number of desired reconstructions to plot.
        
    Returns: 
        None, but saves figures to desired path.
        
    Prereqs:
        1 <= n <= batch size
    """
    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(12, 9))
    fig.suptitle(f'Predictions After Epoch {epoch}', fontsize=16, fontweight='bold')

    test_images = next(iter(test_dataset))
    # first, re-shape the input for compatibility with nn.Conv2d:
    test_images = test_images.unsqueeze(1).to(device)
    loss, reconstructed_images, _ = model(test_images)
    
    with torch.no_grad(): # turn off gradient computations
        for i in range(n):

            # Original (test) images
            axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')
      
            # Reconstructed (by VQVAE) images
            axes[1, i].imshow(reconstructed_images[i].cpu().squeeze().clamp(0,1), cmap='gray')
            axes[1, i].set_title(f'Reconstruction {i+1}', fontweight='bold')
            axes[1, i].axis('off')

    plt.suptitle(f'VQVAE Reconstructions - Epoch {epoch+1}', fontsize=14)
    plt.tight_layout()
    output_file = os.path.join(plot_save_path, f'VQVAE_recon_epoch{epoch}.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
   
    model.train()  # Switch back to training mode

########################################################################
##### Visualise Loss on the reconstructions (as measured by SSIM) ######
########################################################################
def plot_val_SSIMs(plot_save_path, ssims):
    """Create plot of loss (SSIM) against epoch number.

    Args: 
        ssims (list): list of average SSIMs for each epoch, computed
            on the validation dataset. 
    """
    plt.figure(figsize=(8, 4))
    plt.plot(ssims, 'bo-', linewidth=2, markersize=8)
    plt.title('SSIMs calculated on validation dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average SSIM')
    plt.grid(True, alpha=0.3)

    output_file = os.path.join(plot_save_path, 'val_SSIMs.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

###########################################################################
##### Visualise VQVAE Training and Validation losses during training ######
###########################################################################
def plot_training_loss(plot_save_path, train_losses, val_losses=None):
    """Create and save plot of loss (VQVAE Loss) against epoch number.

    Args: 
        train_losses (list or np.array): list of training losses
        val_losses (list or np.array, optional): list of validation losses,
            which will be plotted on the same figure. 
        plot_save_path (str): directory into which to save the plot. 

    Output:
        None, but saves the figure in the specified directory.

    Prereqs:
        len(train_losses) = len(val_losses) if val_losses is not None
    """
    print('Creating final plot of training and validation losses:')
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, 'bo-', linewidth=2, markersize=8)
    if val_losses is not None:
        plt.plot(val_losses, 'r^-', linewidth=2, markersize=8)
    plt.title('VQVAE Loss over Training Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('VQVAE Loss')
    if val_losses is not None:
        plt.legend(['Training','Validation'])
    plt.grid(True, alpha=0.3)

    output_file = os.path.join(plot_save_path, 'train_val_losses.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

####################################################
##### Plot PixelCNN Training Loss over Epochs ######
####################################################
def plot_PixelCNN_loss(plot_save_path, train_losses, val_losses=None):
    """Create and save plot of loss (VQVAE Loss) against epoch number.

    Args: 
        train_losses (list or np.array): list of training losses
        val_losses (list or np.array, optional): list of validation losses,
            which will be plotted on the same figure. 
        plot_save_path (str): directory into which to save the plot. 

    Output:
        None, but saves the figure in the specified directory.

    Prereqs:
        len(train_losses) = len(val_losses) if val_losses is not None
    """
    print('Creating final plot of training losses for PixelCNN:')
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, 'bo-', linewidth=2, markersize=8)
    if val_losses is not None:
        plt.plot(val_losses, 'r^-', linewidth=2, markersize=8)
    plt.title('PixelCNN Loss over Training Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('PixelCNN Loss')
    if val_losses is not None:
        plt.legend(['Training','Validation'])
    plt.grid(True, alpha=0.3)

    output_file = os.path.join(plot_save_path, 'PixelCNN_train_val_losses.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

######################################################
##### Plot the PixelCNN-generated HipMRI Images ######
######################################################
def plot_generated_images(generated_images, epoch: int):
    """Show model predictions after a specific epoch.
    
    Args:
        generated_images: from function generate_images in train.py
        epoch (int): number of epochs
        
    Returns: 
        None, but saves figures to desired path.
    """
    n = generated_images.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(12, 9))
    fig.suptitle(f'Generations After Epoch {epoch}', fontsize=16, fontweight='bold')
    
    for i in range(n):

        image = generated_images[i]
        image = image.squeeze(0)
        axes[i].imshow(image.detach().cpu().numpy(), cmap='gray')
        axes[i].set_title(f'Image number {i+1}', fontweight='bold')
        axes[i].axis('off')

    plt.suptitle(f'VQVAE Generations - Epoch {epoch+1}', fontsize=14)
    plt.tight_layout()
    output_file = os.path.join(plot_save_path, f'VQVAE_gen_epoch{epoch}.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
