# This file contains the source code of the components of the VQVAE model. 
# Each component will be implemented as a class or a function.

####################
#### libraries #####
####################
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

#################
#### device #####
#################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
    
######################################################
##### define residual blocks for encoder/decoder #####
######################################################
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
        # The below line defines a Residual block for each layer
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


####################
##### Encoder ######
####################
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
        
#############################
##### Vector Quantiser ######
#############################
class VectorQuantiser(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """Vector Quantiser module. 
        
        This class creates and manages the embedding matrix (= codebook
        embedding), that stores all the possible discrete embeddings. 

        Args:
            num_embeddings (int): number of embedding vectors (K, in
                the paper by van den Oord et al.)
            embedding_dim (int): dimensionality of each latent 
                embedding vector (D, in the paper by van den Oord et al.)
            commitment_cost (float): weighting factor for the commitment 
                loss term. (=beta in the paper by van den Oord et al.)
        """
        super(VectorQuantiser, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        # create the embedding table/matrix (aka codebook) 
        self._embedding = nn.Embedding(self._num_embeddings, 
                                       self._embedding_dim)
        
        # initialise the embedding weights, using a uniform
        # distribution - to avoid starting with any bias.  
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
        N, C, H, W = inputs.shape
        # where N=batch size, C=embedding_dim, H,W are image height,width.
        # (N, C, H, W) -> (N, H, W, C)
        input_reshaped = inputs.permute(0, 2, 3, 1).contiguous() 
        # (N, H, W, C) -> (N*H*W, C)
        input_flattened = input_reshaped.reshape(N*H*W, self._embedding_dim)
            
        dist = torch.sum(input_flattened**2, dim=1, keepdim=True)+ \
              torch.sum(self._embedding.weight**2, dim=1) - \
                2*torch.matmul(input_flattened, self._embedding.weight.t())
            
        # Nearest neighbour lookup, find which codebook embedding to use
        e_index = torch.argmin(dist, dim=1).unsqueeze(1)
        encodings = torch.zeros(e_index.shape[0], self._num_embeddings, device=device)
        encodings.scatter_(1, e_index, 1)  
        
        # Quantise and unflatten to get the Input to the decoder. 
        # Multiply encodings by their respective weights. Also reshape
        # the quantised tensor to [N,H,W,C] for consistency, and compatibility
        # with F.mse_loss later on.
        quantised = torch.matmul(encodings, self._embedding.weight).view(N, H, W, C)               

        # using torch.detach() method as the 'stopgradient' operator. This
        # helps to constrain its operand to be a non-updated constant. 
        # the output of the encoder is represented here by input_reshaped.
        codebook_loss = F.mse_loss(input_reshaped.detach(), quantised)
        commitment_loss = F.mse_loss(input_reshaped, quantised.detach())
        VQ_layer_losses = codebook_loss + self._commitment_cost*commitment_loss

        # [N, H, W, C] -> [N, C, H, W]
        quantised = quantised.permute(0,3,1,2)

        # Straight Through Estimator - to allow gradient backprop
        quantised = inputs + (quantised - inputs).detach()

        return VQ_layer_losses, quantised, e_index.view(N, H, W)

####################
##### Decoder ######
####################
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
 
        # output from deconv2 is 1 channel, and sigmoid preserves this number
        sigmoid_activation = nn.Sigmoid()
        x = sigmoid_activation(x)

        return x

#######################
##### Full Model ######
#######################
class Model(nn.Module):
    """Full VQVAE Model class
    
    Makes use of each of the Encoder, VectorQuantiser, and
    Decoder components to construct the final VQVAE structure"""
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
        
        First create an encoder class, then use an additional 
        convolution step prior to passing encoder output through
        the vector quantiser. From the VQ layer also obtain the 
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
    
    def encode(self, x):
        """Return the encoded values for a given input/image.

        This involves a pass through the encoder, followed by a 
        pass through the VectorQuantiser.
        """
        x = self._encoder(x)
        x = self._pre_VQ_conv(x)
        _, _, encodings = self._VQ(x) # [N,H,W]

        return encodings
    
    def decode(self, x):
        """Return the decoded value given an input discrete codes.
        
        Args: 
            x: discrete embeddings
        """
        x = self._decoder(x)
        return x
        

######################################################
##### Define Masked Convolutions (for PixelCNN) ######
######################################################
class MaskedConv2d(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels,
                 kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding)
        weight = self.conv.weight.data.clone()
        self.register_buffer('mask', torch.ones_like(weight))
        N, C, H, W = self.conv.weight.size()

        self.mask[:,:,H//2, W//2 + (mask_type=='B'):] = 0
        self.mask[:,:,H//2 + 1] = 0

    def forward(self,x):
        return F.conv2d(x, self.conv.weight*self.mask, self.conv.bias, 
                        stride=self.conv.stride, padding=self.conv.padding)

############################
##### Define PixelCNN ######
############################
class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 kernel_size, n_layers, hidden_channels=64):
        super().__init__()
        
        # Mask A applies only to the first convolutional layer
        self.input_conv = MaskedConv2d('A',in_channels=embedding_dim,
                                       out_channels=hidden_channels, 
                                       kernel_size=7, padding=kernel_size//2)
        self.hidden_layers = nn.ModuleList([
            MaskedConv2d('B', in_channels=hidden_channels,
                         out_channels=hidden_channels, kernel_size=kernel_size,
                         padding=kernel_size//2)
                         for _ in range(n_layers)
        ])
        self.output_conv = nn.Conv2d(hidden_channels, num_embeddings, 1)

    def forward(self, x):
        x = F.relu(self.input_conv(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        logits = self.output_conv(x)
        return logits 
        
##########################
##### Define Models ######
##########################
VQVAE_Model = Model(num_hidden=num_hidden, num_resid_layers=num_resid_layers, 
                 num_resid_hiddens=num_resid_hiddens, 
                 num_embeddings=num_embeddings, 
                 embedding_dim=embedding_dim, 
                 commitment_cost=commitment_cost)

VQVAE_Model = VQVAE_Model.to(device)

PixelCNN_Model = PixelCNN(num_embeddings=num_embeddings,
                          embedding_dim=embedding_dim, 
                        kernel_size=kernel_size, 
                        n_layers=n_layers)

PixelCNN_Model = PixelCNN_Model.to(device)
