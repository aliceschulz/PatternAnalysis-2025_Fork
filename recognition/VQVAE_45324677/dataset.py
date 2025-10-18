# This file contains the data loader for loading and preprocessing the data, as 
# well as various other helper functions for plotting and loading. 
# Data is sourced from the HipMRI Study on Prostate Cancer, and available in
# the COMP3710 folder on the Rangpur cluster. 

import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils
from torch.utils.data import DataLoader

print('Defining data directories...')
dir_test = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
dir_train = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
dir_validation = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"

# Hyperparameters
batch_size = 32

# convert to one-hot encoded: for categorical data. 
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Converts a 2D array of categorical labels to a one-hot encoded 3D array.

    This code was provided by Shakes Chandra in the Open Source Project 
        documentation: COMP3710_Report_v1.64_Final.pdf
    
    Args: arr (np.ndarray): a 2D array of categorical labels
          dtype: data type of the output array
          
    Returns: np.ndarray: a 3D one-hot encoded array
    """
    channels = np.unique(arr) # returns the unique values in the array
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype) # creates a 3D zero-vector
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1 # changes to 1, where the label matches
                                    # at the corresponding channel

    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, early_stop=False):
    """Function to load 2D medical images from Nifti files. 

    This code was provided by Shakes Chandra in the Open Source Project 
        documentation: COMP3710_Report_v1.64_Final.pdf

    Args: imageNames: list of str: list of file paths to Nifti images
          normImage (bool): whether to normalise the image. Normalises 
            to mean 0, std 1 over entire image (subtract mean, divide by std)
          categorical (bool): whether the images are categorical 
            labels (if so, one-hot encode them)
          dtype: data type of output array
          getAffines:
          early_stop (bool): whether to stop loading pre-maturely, 
            leaves arrays mostly empty, for quick loading and testing scripts 

    Returns: np.ndarray: array of loaded images. 
    """
    affines = []
    num_mismatch = 0

    # get fixed size of images - how many images
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0]
        # this removes the extra dims if there are extra,
        # which sometimes there are.
    if categorical: # use one-hot encoding
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if niftiImage.shape != first_case.shape:
            print(f"Image shape mismatch for {inName}, skipping this image.")
            num_mismatch += 1
            continue
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] #sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        print(f"Number of dimension-mismatched images: {num_mismatch}")
        return images, affines
    else: 
        print(f"Number of dimension-mismatched images: {num_mismatch}")
        return images
    
img_names = os.listdir(dir_train) # provides a list of all filenames.
img_paths = [os.path.join(dir_train, name) for name in img_names]
image_training_data = load_data_2D(img_paths)

img_names = os.listdir(dir_test)
img_paths = [os.path.join(dir_test, name) for name in img_names]
image_test_data = load_data_2D(img_paths)

img_names = os.listdir(dir_validation)
img_paths = [os.path.join(dir_validation, name) for name in img_names]
image_validation_data = load_data_2D(img_paths)

# define a function for making a few plots of the original image:
def plt_original_imgs(image, save_path, index):
    """Plot original HipMRI Study on Prostate Cancer images.

    Args: 
        image: np.ndarray, image to be plotted. Often will be output from 
               load_data_2D function.
        save_path: str, directory to the save location
        index: int, image index. 

    Returns:
        None, but saves plot to specified location.
    """
    if image.ndim == 3:
        image = image.squeeze()

    plt.imshow(image, cmap='gray')
    plt.title('Image of HipMRI Study on Prostate Cancer')
    plt.axis('off')

    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'HipMRI_img_{index}.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

for i in range(10):
    plt_original_imgs(
            image=image_training_data[i],
            save_path='/home/Student/s4532467/plots',
            index=i
            )
    

# next, implement the proper data loader to be implemented with the model.
training_loader = DataLoader(image_training_data, 
                             batch_size=batch_size,
                             shuffle=True)


test_loader = DataLoader(image_test_data, 
                         batch_size=batch_size,
                         shuffle=False)

validation_loader = DataLoader(image_validation_data,
                               batch_size=32,
                               shuffle=True)
