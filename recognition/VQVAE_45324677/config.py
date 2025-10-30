# Configuration file
# Specifies specific constants 

#######################
##### dataset.py ######
#######################

batch_size = 32 
plot_images = False # set to True, to create some plots of the original HipMRI images.

#######################
##### plotting.py #####
#######################

plot_save_path = "/home/Student/s4532467/plots"

######################
##### modules.py #####
######################

channels = 1
num_hidden = 128
num_resid_layers = 3
num_resid_hiddens = 64
num_embeddings = 256
embedding_dim = 32
commitment_cost = 0.5
kernel_size = 7 # for PixelCNN
n_layers = 5 # for PixelCNN

####################
##### train.py #####
####################

num_epochs = 20
learning_rate = 1e-4 # for both VQVAE and PixelCNN
visualise_every = 5 # for VQVAE reconstructions
plot_metrics = True
num_epochs_pixelCNN = 50