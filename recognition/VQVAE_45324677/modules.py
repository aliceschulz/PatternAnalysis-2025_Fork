# This file contains the source code of the components of the VQVAE model. 
# Each component will be implemented as a class or a function.

# we will need:

# 1) a Vector Quantiser class. 
# 2) An encoder class.
# 3) A decoder class. 
# 4) A Model class that puts the components together.
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
# - num_embeddings (number of embedding vectors K in the codebook)
#   -> increasing this increases the capacity in the info-bottleneck
#   Sources say there is a trade-off between num_embeddings and
#   embedding_dim. 

# - embedding_dim (dimensionality D of each embedding vector)
#   -> D does not change the capacity in the information-bottleneck

# - commitment_cost (beta, the weighting factor for commitment loss term)

# - num_resid_layers (number of residual layers in encoder/decoder residual
#   stacks. Sources online suggest a number between 3-9, with greater 
#   reconstruction quality generally obtained with a larger number. 

# - num_hidden. The number of channels in the hidden layers of the encoder
#   and decoder. Defines the dimensionality of the feature maps, and dictates 
#   the model's capacity to learn and represent complex patterns in the data 
#   by controlling the size of the intermediate representations. Sources say
#   that generally a larger value can lead to better reconstruction quality. 

# - num_resid_hiddens. Analagous to num_hidden, but for the residual stack/blocks. 
#   So the number of channels in the hidden layers of each residual block. 
#   May be a good idea to gradually increase the number of channels as the 
#   network gets deeper. 

import torch
import torch.nn as nn
import torch.nn.functional as F

# set device to allow GPU computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyperparameters
channels = 1
    
# Encoder and decoder architecture, which will be based on 
# residual blocks, akin to ResNet - as many implementations in 
# practice use this method. 
class Residual(nn.Module):
    """A class to define a single residual block.
    
    This is a generic residual block that can be re-used. Outputs
    from each part are passed along each module in a sequential manner.
    """
    def __init__(self, in_channels, num_hidden, num_resid_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_resid_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_resid_hiddens,
                      out_channels=num_hidden,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    """Defines a stack of residual blocks."""
    def __init__(self, in_channels, num_hidden, num_resid_layers,
                  num_resid_hiddens):
        super(ResidualStack, self).__init__()
        self._num_resid_layers = num_resid_layers
        # The below line defines a Residual block for each layer (given 
        # by the number of specified residual layers). Each of these
        # blocks is then stored in a ModuleList. 
        # num_resid_layers is the number of Residual blocks in the 
        # residual stack. 
        self._layers = nn.ModuleList([Residual(in_channels, 
                                               num_hidden, 
                                               num_resid_hiddens)
                             for _ in range(self._num_resid_layers)])

    def forward(self, x):
        for i in range(self._num_resid_layers):
            x = self._layers[i](x) # each element in self._layers is 
                                    # a residual block. The loop performs
                                    # the forward pass through each block, 
                                    # sequentially, and then the next line 
                                    # returns the activation function ReLU
                                    # applied to the final output. 
        return F.relu(x)


# Next, define the encoder and decoder classes
class Encoder(nn.Module):
    """Defines the Encoder module of the VQVAE model.
    
    The encoder takes, as input, HipMRI image data and encodes
    this via a CNN framework into a discrete latent representation.
    
    While the VQ module determines the closest embedding vector, and 
    sets the encoder output to this, the encoder is responsible for
    the series of convolutional downsamplings and extraction of 
    features from the input data.
    """
    def __init__(self, in_channels, num_hidden, num_resid_layers, 
                 num_resid_hiddens):
        super(Encoder, self).__init__()

        self._conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hidden//2,
                                kernel_size=4,
                                stride=2,
                                padding=1)

        self._conv2 = nn.Conv2d(in_channels=num_hidden//2,
                                out_channels=num_hidden,
                                kernel_size=4,
                                stride=2,
                                padding=1)

        self._conv3 = nn.Conv2d(in_channels=num_hidden,
                                out_channels=num_hidden,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hidden,
                                             num_hidden=num_hidden,
                                             num_resid_layers=num_resid_layers,
                                             num_resid_hiddens=num_resid_hiddens)
        
    def forward(self, inputs):
        """Forward pass for encoder.
        """
        x = F.relu(self._conv1(inputs))
        x = F.relu(self._conv2(x))

        x = self._residual_stack(self._conv3(x))
        return x
        
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
        # a fixed dictionary of size num_embeddings, and size embedding_dim
        # of each embedding vector. 
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
            - re-shaping of input to the appropriate shape (note: 
                the 'inputs' come from the encoder, they are not
                directly the images themselves)
            - calculation of distance between point and 
            embedding vector
            - determining which is the closest embedding vector
            (and therefore setting the encoding to this)
            - loss function term calculation
        """
        # first, we are required to reshape and flatten the encoded inputs. 
        # from (N, C, H, W) -> (N*H*W, C)
        N, C, H, W = inputs.shape
        # where N=batch size, C=channels, H,W are height,weight of image.
        # Note that C (num channels) is equal to the embedding dim, due
        # to the pre-VQ convolution layer after the encoder that makes 
        # it such. 
        # Desired ordering is N, H, W, C, so enter this as such:
        input_reshaped = inputs.permute(0, 2, 3, 1).contiguous() # [N, H, W, C]
        # flatten the first three dims:
        input_flattened = input_reshaped.reshape(N*H*W, self._embedding_dim) #[N*H*W, C]
            
        # Calculate distances
        # note that the original van den Oord paper used argmin on 
        # the regular l2 norm, but here we argmin the squared l2 norm,
        # as it is equivalent and slightly easier to compute. 
        dist = torch.sum(input_flattened**2, dim=1, keepdim=True)+ torch.sum(self._embedding.weight**2, dim=1)- 2*torch.matmul(input_flattened, self._embedding.weight.t())
            
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
        print(f"encodings value counts: {torch.unique(encodings, return_counts=True)}")
        print(f"encodings shape: {encodings.shape}")
        
        # Quantise and unflatten to get the Input to the decoder. 
        # The encodings are multiplied by their respective weights. Also reshape
        # the quantised tensor to [N,H,W,C] for consistency, and compatibility
        # with F.mse_loss later on.
        quantised = torch.matmul(encodings, self._embedding.weight).view(N, H, W, C)

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
        # using mse_loss, as squared euclidian norm is equivalent to thiS.
        # z_e is the output of the encoder, and here is represented by input_reshaped.
        codebook_loss = F.mse_loss(input_reshaped.detach(), quantised)
        commitment_loss = F.mse_loss(input_reshaped, quantised.detach())
        VQ_layer_losses = codebook_loss + self._commitment_cost*commitment_loss

        # when returning quantised, reshape back to [N, C, H, W]
        quantised = quantised.permute(0,3,1,2)

        # Straight Through Estimator - to allow differentiability, so that
        # the gradient backpropagations skip the non-differentiable vector
        # quantised part. 
        quantised = inputs + (quantised - inputs).detach()

        return VQ_layer_losses, quantised, e_index.view(N, H, W)

class Decoder(nn.Module):
    """Defines the Decoder module of the VQVAE model.
    
    The decoder takes as input the discrete latent embeddings,
    and through a series of convolutional upsampling, attempts
    to reconstruct the original input.
    """
    def __init__(self, in_channels, num_hidden, num_resid_layers, 
                 num_resid_hiddens):
        """
        Args:
            in_channels (int): number of input channels
            num_hidden (int): number of hideen??
            num_resid_layers (int):
            num_resid_hiddens (int):
        """
        super(Decoder, self).__init__()

        self._conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hidden,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hidden,
                            num_hidden=num_hidden,
                            num_resid_layers=num_resid_layers,
                            num_resid_hiddens=num_resid_hiddens)
        
        self._deconv1 = nn.ConvTranspose2d(in_channels=num_hidden,
                                           out_channels=num_hidden//2,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        
        self._deconv2 = nn.ConvTranspose2d(in_channels=num_hidden//2,
                                           out_channels=1,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        
    def forward(self, inputs):
        """Forward pass for the decoder.

        Passes inputs through the first convolutional layer, then
        the residual stack, followed by the final deconvoltuion 
        layers and an appropriate activation layer. 
        """
        x = self._conv1(inputs)
        x = self._residual_stack(x)

        x = F.relu(self._deconv1(x))
        x = self._deconv2(x)
 
        #output channels from deconv2 is 1 channel
        # sigmoid should preserve this number
        sigmoid_activation = nn.Sigmoid()
        x = sigmoid_activation(x)

        return x
        
# Next, define the Model class that makes use of each
# of these components to construct the final VQVAE structure.
class Model(nn.Module):
    def __init__(self, num_hidden, num_resid_layers, 
                 num_resid_hiddens, num_embeddings, 
                 embedding_dim, commitment_cost):
        super(Model, self).__init__()

        self._num_embeddings = num_embeddings

        self._encoder = Encoder(in_channels=channels, num_hidden=num_hidden, 
                                num_resid_layers=num_resid_layers, 
                                num_resid_hiddens=num_resid_hiddens)
        
        # the below layer ensures that the right number of channels feed into 
        # the VQ module. Otherwise, the number of out channels from the 
        # Encoder module may not match the expected number of in channels
        # for the VQ module. 
        self._pre_VQ_conv = nn.Conv2d(in_channels=num_hidden,
                                      out_channels=embedding_dim,
                                      kernel_size=1, stride=1)
        
        self._VQ = VectorQuantiser(num_embeddings, embedding_dim,
                                   commitment_cost)
        
        self._decoder = Decoder(in_channels=embedding_dim, 
                                num_hidden=num_hidden,
                                num_resid_layers=num_resid_layers,
                                num_resid_hiddens=num_resid_hiddens)
        
    def forward(self, x):
        """Forward pass through the entire VQVAE model as a whole.
        
        First we create an encoder class, then use an additional 
        convolution step prior to passing encoder output through
        the vector quantiser. From the VQ layer we also obtain the 
        codebook loss and commitment loss, the decoder input, 
        and the encodings. Then, the decoder input is fed to the 
        decoder, and from here the reconstructed version of the 
        original input is obtained.
        """
        z = self._encoder(x)
        z = self._pre_VQ_conv(z)

        loss, quantised, encodings = self._VQ(z)
        x_reconstructed = self._decoder(quantised)

        return loss, x_reconstructed, encodings

    def get_embeddings(self):
        """Return number of embeddings"""

        return self._num_embeddings
        

# currently assumes the same number of residual layers in both 
# the encoder and the decoder. 
# Preliminary starting values for hyperparameters are here:
VQVAE_Model = Model(num_hidden=128, num_resid_layers=3, 
                 num_resid_hiddens=64, num_embeddings=256, 
                 embedding_dim=32, commitment_cost=0.5)
# utilised a starting value of commitment cost (Beta) = 0.25, according
# to the value utilised in the van den Oord et al. paper. 
VQVAE_Model = VQVAE_Model.to(device)
