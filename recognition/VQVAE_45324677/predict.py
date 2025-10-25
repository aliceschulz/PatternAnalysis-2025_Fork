# This script is to show example usage of the trained model.
# Results/visualisations will be printed where applicable.

# presumably this is where the final SSIM accuracy and final loss will be evaluated?
# also include the FINAL output plots - i.e. not just ones during training
# presumably if I wish to plot/view some things during training, that 
# will necessarily have to be in the training loop?
import torch
import time
from skimage.metrics import structural_similarity as ssim
from dataset import *
from train import *


# reqs: shape(img1) = shape(img2)
# data_range: the data range of the input image (difference between max and min possible vals)
#   by default, this is estimated from the image data type. this estimate may be wrong for 
#   floating point image data. recommended to pass this scalar value explicitly. 
# returns: mssim (float): the meas structural similarity index over the image.
# 
# # check final testing loss, and final SSIMs
# # assuming model has already been fully trained
# 

# Test the model
print("Begin testing")
start = time.time() #time generation
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

        for i in range(N): # check how the indexing should be done
            inputs_i = inputs[i].squeeze(0).cpu().numpy()
            # torch.Size([1,256,128]) -> torch.Size([256,128]) -> [256,128]
            recon_i = test_data_recon[i].squeeze(0).cpu().numpy()
            ssim_img = ssim(inputs_i, recon_i,
                            data_range=recon_i.max() - recon_i.min())
            #print(f'ssim_img for img {i}, batch {batch_id}: {ssim_img}')
            ssims.append(ssim_img)

    print(f"""Final test Loss: {running_test_loss}, 
          Average Test Loss: {running_test_loss / len(test_loader.dataset)},
          Average SSIM across {len(test_loader.dataset)} images: {np.mean(ssims)}""")

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")