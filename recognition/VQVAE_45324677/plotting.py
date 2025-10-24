# A script to separate out the specific plotting functions.
# Acknowledgement: This script was adapted from Dr Wei Dai
# "UNet_segmentation_code_demo.ipynb" 

import matplotlib.pyplot as plt
import torch
import os

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_epoch_reconstructions(model, test_dataset, epoch, n=10):
    """Show model predictions after a specific epoch.
    
    Args:
        model: of class nn.Module. 
        test_dataset: DataLoader class, containing the test images.
        epoch: int. Specified within training loop; provides an epoch
            number that can be included in plot titles. 
        n: int, number of desired reconstructions to plot.
        
    Output: 
        None, but saves figures to desired path."""
    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(12, 9))
    fig.suptitle(f'🎯 Predictions After Epoch {epoch}', fontsize=16, fontweight='bold')

    test_images = next(iter(test_dataset))
    # first, re-shape the input for compatibility with nn.Conv2d:
    test_images = test_images.unsqueeze(1).to(device)
    loss, reconstructed_images, encodings = model(test_images)
    
    with torch.no_grad(): # turn off gradient computations
        for i in range(n):

            #loss, reconstructed_image, encodings = model(image.unsqueeze(1).to(device))

            # Denormalize image for visualization
            #img_show = denormalize_image(image)

            # Show original image
            print('** Test images ** (for plotting)')
            print(f"Test image {i}: min={test_images[i].min()}, max={test_images[i].max()}")
            axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')
            #if i == 0:
            #    axes[0, i].set_ylabel('Original', fontsize=12)

            # Reconstructed
            print('** Reconstructed images **')
            print(f"Reconstructed image {i}: min={reconstructed_images[i].min()}, max={reconstructed_images[i].max()}")
            axes[1, i].imshow(reconstructed_images[i].cpu().squeeze().clamp(0,1), cmap='gray')
            axes[1, i].set_title(f'Reconstruction {i+1}', fontweight='bold')
            axes[1, i].axis('off')
            #if i == 0:
            #    axes[1, i].set_ylabel('Reconstructed', fontsize=12)

    plt.suptitle(f'VQVAE Reconstructions - Epoch {epoch+1}', fontsize=14)
    plt.tight_layout()
    output_file = os.path.join(plot_save_path, f'VQVAE_recon_epoch{epoch}.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
   
    model.train()  # Switch back to training mode

# Quick visualization of loss
def plot_loss(losses):
    """Create plot of loss (SSIM) against epoch number."""
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'bo-', linewidth=2, markersize=8)
    plt.title('SSIM'), fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
