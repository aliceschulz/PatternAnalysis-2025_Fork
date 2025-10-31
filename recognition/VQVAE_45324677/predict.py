# This script is to show example usage of the trained model.
# Results/visualisations will be printed where applicable.


######################
##### Libraries ######
######################
import torch
import time
from skimage.metrics import structural_similarity as ssim
from dataset import *
from train import * # runs training script


#######################
##### Test VQVAE ######
#######################
print("Begin testing")
start = time.time() #time tracking
VQVAE_Model.eval()
ssims = []
with torch.no_grad():
    running_test_loss = 0
    for batch_id, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(1)

        N, C, H, W = inputs.shape 

        test_loss, test_data_recon, _ = VQVAE_Model(inputs)
        test_loss = full_VQVAE_loss(test_loss, test_data_recon, inputs)
        running_test_loss += test_loss.item() * inputs.size(0)
        # update running test loss, multiply each avg batch loss by batch size.

        for i in range(N):
            # [1,H,W] -> [H,W]
            inputs_i = inputs[i].squeeze(0).cpu().numpy()
            recon_i = test_data_recon[i].squeeze(0).cpu().numpy()
            ssim_img = ssim(inputs_i, recon_i,
                            data_range=recon_i.max() - recon_i.min())
            
            ssims.append(ssim_img)

    print(f"""Final test Loss: {running_test_loss}, 
          Average Test Loss: {running_test_loss / len(test_loader.dataset)},
          Average SSIM across {len(test_loader.dataset)} images: {np.mean(ssims)}""")

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

##########################
##### Test PixelCNN ######
##########################
print("Now testing the PixelCNN...")
start = time.time()
PixelCNN_Model.eval()
VQVAE_Model.eval() # using already trained VQVAE
pixel_test_losses = []
with torch.no_grad():
    running_test_loss = 0.0
    for batch_id, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(1)
        
        latents = VQVAE_Model.encode(inputs).to(device)
        latents_embedded = VQVAE_Model._VQ._embedding(latents).permute(0,3,1,2).float()
        latents_embedded = latents_embedded.to(device)
                
        logits = PixelCNN_Model(latents_embedded)
        loss = F.cross_entropy(logits, latents)

        running_test_loss += loss.item() * inputs.size(0)
        # update running test loss, multiply each avg batch loss by batch size.

    print(f"""Final test Loss: {running_test_loss}, 
          Average Test Loss: {running_test_loss / len(test_loader.dataset)}""")
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total for the PixelCNN")