############################################################
#
#This script contains all files used to preprocess data into patches or generate features out of an image/patch
#as well as preprocess the ground truth images. 
#
############################################################


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import os
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, random_split

def parse_mask_to_binary(input_mask, grayscale_threshold = 0.5):
    """
    Function that takes an RGB mask and outputs the corresponding binary mask
    """
    binary_mask = np.where(color.rgb2gray(input_mask) == 0, 0, 1)
    return binary_mask

def img_to_patches(img, number_of_patches, print_=False):
    """
    Function that takes full size image and outputs a list of the image's patches
    number of patches divides the image into number of those patches
    """
    patches_per_dimension = np.sqrt(number_of_patches)
    #assert img.shape[0]%patches_per_dimension== 0 & img.shape[1]%patches_per_dimension == 0

    patch_width, patch_height = (int(img.shape[0]/patches_per_dimension), int(img.shape[1]/patches_per_dimension))
    list_of_patches = []

    for i in range(number_of_patches):
        x = int(i/patches_per_dimension)*patch_width
        y = int((i%patches_per_dimension)*patch_height)
        list_of_patches.append(
            img[x:x+patch_width, y:y+patch_height]
        )
    if (print_): print("patch dim", patch_width, patch_height)
    return list_of_patches

def patch_to_features(patch, n_features=20):
    """
    Function that takes in 2d patch of an RGB image as an input and outputs a feature vector for this patch
    """
    #supposing patches come in RGB
    feature_vector = []
    patch_hsv = color.rgb2hsv(patch)
    concatenated_channels = np.concatenate([patch, patch_hsv], axis=2)
    n_channels = concatenated_channels.shape[2]

    for cdx in range(n_channels): #With expected R,G,B,H,S,V channels this gives us 12 features per patch already
        feature_vector.append(np.mean(concatenated_channels[:, :, cdx].flatten()))
        feature_vector.append(np.std(concatenated_channels[:, :, cdx].flatten()))

    #Texture features

    #Create bank of gabor filters

    #Convolution and take middle pixel?? bruh momento

    return feature_vector

def groundtruth_patch_to_label(patch, foreground_threshold = .5):
    """
    Takes patch from mask and computes average label value
    likeliness of patch to be road or background
    """
    return int(np.mean(patch.flatten()) > foreground_threshold)

def predicted_patch_labels_to_mask(labels, output_shape): #array 1d, tuple (2)
    """
    Takes array of patch labels and converts it into binary mask :)
    """
    mask = np.zeros(output_shape)

    patches_per_dimension = np.sqrt(len(labels))
    patch_width, patch_height = (int(output_shape[0]/patches_per_dimension), int(output_shape[1]/patches_per_dimension))
    for ldx, l in enumerate(labels):
        x = int(ldx/patches_per_dimension)*patch_width
        y = int((ldx%patches_per_dimension)*patch_height)

        mask[x:x+patch_width, y:y+patch_height] = l

    return mask

def img_to_patches_fixed(img, patch_size, print_=False):
    """
    Function that takes full size image and outputs a list of the image's patches
    with a fixed patch_size, needs a square image, divisible by patch size
    """

    patches_per_dimension = int((img.shape[0]/patch_size))
    number_of_patches = int((img.shape[0]/patch_size)**2)
    list_of_patches = []

    for i in range(number_of_patches):
        x = int(i/patches_per_dimension)*patch_size
        y = int((i%patches_per_dimension)*patch_size)
        list_of_patches.append(
            img[x:x+patch_size, y:y+patch_size]
        )
    if (print_): print("patch dim", patch_size)
    return list_of_patches

def PatchPred(prediction):
    return np.array(torch.argmax(prediction, dim = 1).detach().cpu().numpy())

class identity:
    def __init__(self) -> None:
        pass
    def transform(self, x):
        return x

def identity_func(x):
    return x.long()

def identity_detach(x):
    return x.detach().cpu().numpy()

def permutate_labels(y):
    return torch.permute(y, (0, 3, 2, 1))