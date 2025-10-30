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


## Description of the Model - VQVAE

Overall, a VQVAE aims to address the 'posterior collapse' problem that may occur in traditional Variational Auto-Encoders (VAEs) by allow a learnt representation of discrete embeddings, rather than modelling a continuous latent space. 

Each component is as follows:

### Encoder

An encoder network outputs discrete codes/embeddings. Similar to a traditional encoder, it functions by taking in the image input data, and through a series of convolutional sampling, results in a different representation. To increase accuracy, this encoder implements residual connections through the use of residual blocks. 

### Vector Quantiser

The Vector Quantiser helps to represent the discrete latent space that is learnt by the VQVAE and that can capture important features of the data in an unsupervised manner. Outputs from the encoder are entered into the vector quantiser, which then uses a nearest neighbour lookup to identify the closest embedding vector within the space. 
Embeddings are represented by the class nn.Embedding, which functions as a simple lookup table that maps an index value to a weight matrix. During training, the parameters of this embedding layer are adjusted, with the embedding matrix (also known as codebook) being updated via backpropagation to minimise the loss function. 

### Vector Quantiser prior

The prior in the VQVAE case is learnt rather than static. In the original VQVAE implementation, an autoregressive prior was used [?].

### Decoder

The decoder takes as input the embedding vector identified by the vector quantiser. Through another series of convolutions, the input is sampled until the final output is produced. Similar to the encoder, residual connections are implemented through the use of stacks of residual blocks. 

### PixelCNN

The PixelCNN is a model that learns to model the prior, and is used for generation. For image generation, it generates every new pixel sequentially, one at a time, and on the basis of (conditioned on) previous pixels it has generated [?]. It uses masked convolutions, in order to set connections to any future pixels to zero, such that the model cannot 'see' these.
To sample from the latent space, the trained PixelCNN is fit over the latent values. 

### Loss

The VQVAE loss is composed of three components, which each have their own interpretation and effect. <description of this>

In terms of measuring reconstruction fidelity, the Structural Similarity Index (SSIM) was used. This is a framework for assessing the similarity and visibility of differences between a 'distorted' image and a reference image, based on the degradation of or change in structural information [?]. It holds a benefit over other metrics as it takes texture and structural information into account, and incorporates perceptual phenomena such as luminance and contrast [?]. In the case of this project, the SSIM is measured in reference to the original uncompressed/unedited HipMRI image. 

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



