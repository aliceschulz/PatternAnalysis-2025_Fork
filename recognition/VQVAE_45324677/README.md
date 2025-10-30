# VQVAE for HipMRI Image Reconstruction

This file includes the code etc. for implementing the topic number 10 of the COMP3710 Project.

Description:
"Create a generative model of the HipMRI Study on Prostate Cancer using the processed 2D slices (2D images) available here with the using a VQVAE or VQVAE2 that has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6."

This folder will contain the solution and its code, and later will contain a summary of the results and model (essentially as a mini-report). 

## Project Overview

This project aims to implement a VQVAE (as described in [1]) to perform image reconstruction of HipMRI images. 

## Dependencies 

Outlined here are the versions used within this VQVAE implementation, and their dependencies.

| Package | Version | Dependencies, if applicable | Usage |
|---|---|---|---|
| python | v3.13.7 | - | - |
| scikit-image | v0.25.2 | numpy>=1.24, scipy>=1.11.4, networkx>=3.0, pillow>=10.1, imageio!=2.35.0,>=2.33, tifffile>=2022.8.12, packaging>=21, lazy-loader>=0.4| SSIM metric |
| tqdm | v4.67.1 | - | Load data function. Progress bars for training loops. |
| nibabel | v5.3.2 | numpy>=1.22, packaging>=20 | Load data function. |
| numpy | v2.1.2 | - | Load data function, linear algebra. |
| packaging | v25.0 | - | - |
| torch | v2.7.1+cu118 | - | Deep learning model. |
| openssl | v3.0.18 | - | tqdm | 
| matplotlib | v3.10.6 | - | Plotting and visualisations. |
| utils | v1.0.2 | - | to_channels function |

## Hyperparameters

| Name | Chosen number for project | Description | Other comments |
|---|---|---|---|
| num_embeddings | 256 | The number of embedding vectors (K) to be used in the codebook. | A larger K leads to increased capacity in the information bottleneck. |
| embedding_dim | 32 | The dimensionality D of each embedding vector. | D does not change the capacity in the information bottleneck. |
| commitment_cost | 0.5 | The weighting factor (beta) for the commitment loss term. | Load data function. Progress bars for training loops. |
| num_resid_layers | 3 | The number of residual layers in the encoder/decoder residual stacks. | A greater reconstruction quality may be obtained with a larger number. |
| num_hidden | 128 | Number of channels in the hidden layers of the encoder/decoder. Defines dimensionality of the feature maps and controls the size of the intermediate representations. | A greater reconstruction quality may be obtained with a larger number. |
| num_resid_hiddens | 64 | Number of channels in the hidden layers of each residual block. | - |

## Description of the Model - VQVAE

Overall, a VQVAE aims to address the 'posterior collapse' problem that may occur in traditional Variational Auto-Encoders (VAEs) by allow a learnt representation of discrete embeddings, rather than modelling a continuous latent space. 

Each component is as follows:
### Encoder

An encoder network outputs discrete codes/embeddings. Similar to a traditional encoder, it functions by taking in the image input data, and through a series of convolutional sampling, results in a different representation by extracting features from the input data. To increase accuracy, this encoder implements residual connections through the use of residual blocks. The encoder makes use of residual stacks, which are a series of residual blocks (the number of which is given by num_resid_layers).

### Vector Quantiser

The Vector Quantiser helps to represent the discrete latent space that is learnt by the VQVAE and that can capture important features of the data in an unsupervised manner. Outputs from the encoder are entered into the vector quantiser, which then uses a nearest neighbour lookup to identify the closest embedding vector within the space. 
Embeddings are represented by the class nn.Embedding, which functions as a simple lookup table that maps an index value to a weight matrix. During training, the parameters of this embedding layer are adjusted, with the embedding matrix (also known as codebook) being updated via backpropagation to minimise the loss function. Embedding weights were initialised using a uniform distribution, to avoid starting with any bias. 

### Vector Quantiser prior

The prior in the VQVAE case is learnt rather than static. In the original VQVAE implementation, an autoregressive prior was used [?].

### Decoder

The decoder takes as input the embedding vector identified by the vector quantiser. Through another series of convolutions, the input is sampled until the final output is produced. Similar to the encoder, residual connections are implemented through the use of stacks of residual blocks. A final sigmoid activation function was utilised as the final layer of the decoder. This is to maintain consistency with the data pre-processing that was utilised; as datasets were standardised to be in [0,1] (which is the range of nn.Sigmoid()).

### PixelCNN

The PixelCNN is a model that learns to model the prior, and is used for generation. For image generation, it generates every new pixel sequentially, one at a time, and on the basis of (conditioned on) previous pixels it has generated [?]. It uses masked convolutions, in order to set connections to any future pixels to zero, such that the model cannot 'see' these.
To sample from the latent space, the trained PixelCNN is fit over the latent values. 

### Loss

The VQVAE loss is composed of three components, which each have their own interpretation and effect. The first term is the codebook loss, which uses the l2 error to move the embedding vectors towards the encoder inputs. The second term is commitment loss. By pushing the encoder to commit to an embedding, it is used to ensure that the volume of the embedding space does not grow arbitrarily. 

In terms of measuring reconstruction fidelity, the Structural Similarity Index (SSIM) was used. This is a framework for assessing the similarity and visibility of differences between a 'distorted' image and a reference image, based on the degradation of or change in structural information [?]. It holds a benefit over other metrics as it takes texture and structural information into account, and incorporates perceptual phenomena such as luminance and contrast [?]. In the case of this project, the SSIM is measured in reference to the original uncompressed/unedited HipMRI image. 

### Optimiser, and optimisation process

Optimisation process utilises the Straight Through Estimator trick. As the latent space is discretised, this step is required to allow differentiability and so that the gradient backpropagations may pass through the non-differentiable vector-quantised component. 

### 

## Data & Preprocessing

Normalisation was performed by standardising each image such that each pixel value falls in [0,1]. It was decided to not standardise each image to a mean of 0 and standard deviation of 1, because upon inspection, the pixel values for each image clearly did not appear normally distributed (on the contrary, they appeared quite non-normal and skewed to the right). This standardisation was carried out on a global basis (using the global max value for standardisation) rather than on a per-image basis. This is because upon inspection, there was a large degree of variation between pixel max values for each image. Additionally, the standardisation was conducted separated across each split of data (train/test/validate), so as to ensure that scaling parameters such as min/max are not mixed between training and test sets (and therefore, to minimise data leakage [2]). 

The train/test/validate split was as follows: 
| Section | Number of Images | Proportion of Total Images |
|---|---|---|
| Train | 11,460 | 11460/(11460+540+660) = 90.52% |
| Test | 540 | 540/(11460+540+660) = 4.27% |
| Validate | 660 | 660/(11460+540+660) = 5.21% |

Each image is of size (256x128). There were however, 60 images within the training dataset that were inconsistent with these dimensions. These were removed and not implemented in the training of the model, owing to the fact that they constitute a very small proportion of the total test size, and also to avoid any unwanted artefacts being introduced with the re-sizing of these images. 

** Justify the training, validation and testing spits of the data ** 

## Usage

i.e. which commands to run

## Results

### Training / Loss Curves

The loss was evaluated using SSIM (structued similarity index) [3], as well as the VQVAE loss. During training, it was evaluated on both the training dataset and the validation set. 
Plots indicate that training and validation loss for the VQVAE decrease with more epochs. The validation loss starts out higher than the training loss, which is to be expected, but then they approach each other. The validation loss also does not start to overtake the training loss, which is a good indicator that the model is not overfitting. 

### Result Demonstration 

In this section, the results of the reconstruction and generation will be shared.

#### Reconstruction

The VQVAE model succeeded in generating high-quality, reasonably clear reconstructions. Between the original image and the reconstructed one, there is clearly still some degree of blurring, but overall the image was clearly comparable to the original. All components of the image appeared in their expected locations, with consistency in brightness, texture, and scale between the two. Early reconstructions (around Epoch 1) were understandably poor, but the quality continuously improved. 

[Epoch1](plots/VQVAE_recon_epoch1.png)


#### Generation

## Other notes/assumptions
To address data leakage, I have assumed that individuals were *not* repeated across the training, test, and validation sets. I.e., the individuals that constitute the training set are completely independent of the individuals that constitute the testing/validation set, and there are no images in the test/validation set that are of individuals that also appeared in the training set. This would constitute a large data leakage risk and may lead to overestimation of accuracy, even if the images themselves are different. 

## References

Sayash Kapoor, Arvind Narayanan,
Leakage and the reproducibility crisis in machine-learning-based science,
Patterns,
Volume 4, Issue 9,
2023,
100804,


Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004, doi: 10.1109/TIP.2003.819861.
<img width="1190" height="110" alt="image" src="https://github.com/user-attachments/assets/45b44f4f-87be-466e-a444-475b49fb7fb8" />



