# This file contains the data loader for loading and preprocessing the data.
# Data is sourced from the HipMRI Study on Prostate Cancer. 
# rangpur directories:

# test: "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
# train: "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
# validate: "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"

import numpy as np
import nibabel as nib
from tqdm import tqdm

print('Defining data directories...')
dir_test = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
dir_train = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
dir_validate = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Converts a 2D array of categorical labels to a one-hot encoded 3D array.
    
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
          normImage (bool): whether to normalise the image in 
            [0,1] (subtract of mean, divide by std)
          categorical (bool): whether the images are categorical 
            labels (if so, one-hot encode them)
          dtype: data type of output array
          getAffines:
          early_stop (bool): whether to stop loading pre-maturely, 
            leaves arrays mostly empty, for quick loading and testing scripts 

    Returns: np.ndarray: array of loaded images. 
    """
    affines = []

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
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else: 
        return images
    
    