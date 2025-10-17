# This file contains the source code of the components of the VQVAE model. 
# Each component will be implemented as a class or a function.

# we will need:

# 1) a Vector Quantiser class. 
# 2) An encoder class.
# 3) A decoder class. 
# a lot of sources recommend or utilise residual connections in the
# encoder blocks and decoder blocks (similar to ResNet), so we will 
# leverage off this and also utilise some residual blocks. 

# Overall, the model is a combination of these components and needs to:
# - Use the encoder to take in the input data and produces an output. 
# - The encoder's output goes into the vector quantiser thing
# - the vector quantiser uses nearest neighbour lookup to find the closest embedding vector e
# - the corresponding embedding vector is then the input to the decoder
# - the decoder produces the final output. 

# Hyperparameters that will need successive tuning:
# - num_embeddings (number of embedding vectors K)
#   -> increasing this increases the capacity in the info-bottleneck
# - embedding_dim (dimensionality D of each embedding vector)
#   -> D does not change the capacity in the information-bottleneck
# - commitment_cost (beta, the weighting factor for commitment loss term)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# set device to allow GPU computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vector Quantiser class
class VectorQuantiser(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """Vector Quantiser module. 
        
        This class creates and manages the embedding matrix (= codebook
        embedding), that stores all the possible discrete embeddings. 

        Args:
            num_embeddings (int): number of embedding vectors (K, in
                the paper by van den Oord et al.?)
            embedding_dim (int): dimensionality of each latent 
                embedding vector (D, in the paper by van den Oord et al.)
            commitment_cost (float): weighting factor for the commitment 
                loss term. (=\beta in the paper by van den Oord et al.)
        """
        super(VectorQuantiser, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        # create the embedding table that stores embeddings of 
        # a fixed dictionary and size:
        # output of Embedding is of size (*,D) where * is the input
        # shape and D = embedding_dim. 
        self._embedding = nn.Embedding(self._num_embeddings, 
                                       self._embedding_dim)
        
        # initialise the embedding weights, using a uniform
        # distribution - to avoid starting with any bias. 
        # This represents the embedding matrix. 
        # Note: 'weight' is a learnable parameter, and during 
        # training these weights should be updated via backpropagation. 
        self._embedding.weight.data.uniform_(-1/self._num_embeddings,
                                             1/self._num_embeddings)
        
        def forward(self, inputs):
            """Defines the forward pass for the Vector Quantiser.
            
            Each forward pass will include: 
                - re-shaping of input to the appropriate shape
                - calculation of distance between point and 
                embedding vector
                - determining which is the closest embedding vector
                (and therefore setting the encoding to this)
                - loss function term calculation
            """
            # first, we are required to flatten the encoded inputs. 
            # from (B, C, H, W) -> (B*H*W, C)
            B, C, H, W = inputs.shape
            # desired ordering is B, H, W, C, so enter this as such:
            input_reshaped = inputs.permute(0, 2, 3, 1)
            # flatten the first three dims:
            input_flattened = input_reshaped.reshape(B*H*W, 
                                                     self._embedding_dim)
            
            # Calculate distances
            # note that the original van den Oord paper used argmin on 
            # the regular l2 norm, but here we argmin the squared l2 norm,
            # as it is equivalent and slightly easier to compute. 
            dist = torch.sum(input_flattened**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2*torch.matmul(input_flattened, self._embedding.weight.t())
            
            # Using the nearest neighbour lookup, evaluate which codebook
            # embedding to use for the given input
            e_index = torch.argmin(dist, dim=1).unsqueeze(1)
            encodings = torch.zeros(e_index.shape[0], self._num_embeddings, device=device)
            encodings.scatter_(1, e_index, 1)  # writes all values from '1' into the 
                                        # encodings, at the indices specified 
                                        # in 'e_index'. The output index for the '1' 
                                        # is specified by its index in '1' for 
                                        # dimension != dim, and by the corresponding
                                        # value in index for dimension = dim. 
        
            # Quantise and unflatten to get the INput to the decoder. 
            # encodings multiplied by their respective weights
            quantised = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)

            # Loss terms:
            # 1): codebook loss. This uses the l2 error to move the 
            #   embedding vectors towards the encoder inputs.
            #   ||sg[z_e(x)] - e ||_2^2 in original van den Oord paper
            #   AKA MS diff bn output of VQ layer, and input to VQ layer

            # 2): commitment loss. This is to ensure the volume of the 
            #   embedding space does not grow arbitrarily - helps the 
            #   encoder to commit to an embedding. 
            #   || z_e(x) - sg[e] ||_2^2 in original van den Oord paper
            #   AKA MS diff bn output of VQ layer, and input to VQ layer                  

            # using torch.detach() method as the 'stopgradient' operator. This
            # helps to constrain its operand to be a non-updated constant. 
            # using mse_loss, as squared euclidian norm is equivalent to this.
            codebook_loss = F.mse_loss(input_reshaped.detach(), quantised)
            commitment_loss = F.mse_loss(input_reshaped, quantised.detach())
            VQ_layer_losses = codebook_loss + self._commitment_cost*commitment_loss

            return VQ_layer_losses, quantised.permute(0,3,1,2), encodings
