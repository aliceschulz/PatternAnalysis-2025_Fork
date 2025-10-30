# VQVAE for HipMRI Image Reconstruction

This file includes the code etc. for implementing the topic number 10 of the COMP3710 Project.

Description:
"Create a generative model of the e HipMRI Study on Prostate Cancer using the processed 2D slices (2D images) available here with the using a VQVAE or VQVAE2 that has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6."

This folder will contain the solution and its code, and later will contain a summary of the results and model (essentially as a mini-report). 

## Project Overview

This project aims to implement a VQVAE (as described in [1]) to perform image reconstruction of HipMRI images. 

## Dependencies 

Outlined here are the versions used within this VQVAE implementation, and their dependencies.

**| Package | Version | Dependencies, if applicable | Usage | **
|---|---|---|
| python | v3.13.7 | - | - |
| scikit-image | v0.25.2 | numpy >= 1.24| SSIM metric |
| Validate | 660 | 660/(11460+540+660) = 5.21% |

## Description of the Model - VQVAE

Overall, a VQVAE aims to address the 'posterior collapse' problem that may occur in traditional Variational Auto-Encoders (VAEs) by allow a learnt representation of discrete embeddings, rather than modelling a continuous latent space. 

Each component is as follows:

### Encoder

An encoder network outputs discrete codes/embeddings. Similar to a traditional encoder, it functions by taking in the image inputs, and through a series of convolutional sampling, results in a different representation.

### Vector Quantiser

Discrete latent space learnt by the VQVAE can capture important features of the data in an unsupervised manner. 

### Vector Quantiser prior

### Decoder

### Loss

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

The loss was evaluated using SSIM (structued similarity index).

### Result Demonstration 

#### Reconstruction

#### Generation

## References

Sayash Kapoor, Arvind Narayanan,
Leakage and the reproducibility crisis in machine-learning-based science,
Patterns,
Volume 4, Issue 9,
2023,
100804,



