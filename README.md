

## TEAM INFORMATION
This project was conducted by : 
- Nicolas Francio
- Louis Leger
- Daphné de Quatrebarbes

## INTRO
This project focuses on segmenting roads from aerial images using various algorithms and optimization techniques. This README provides essential information on file organization and usage to ensure a smooth experience.

# PROJECT STRUCTURE 

## DATA 
The data sets required to train and test the prediction models we have implemented can be found on [AIcrowd][(https://www.aicrowd.com/challenges/epfl-machine-learning-project-1](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation))
Functions for submission are provided and located in the file import.
Note To run our code, please download these files and put them in the same folder as our code files.

## SCRIPTS FILE 
The scripts file is composed of python files defining our models, training methods, preprocessing, evaluation metrics and methods, data augmentation and imports. 

## BEST MODEL PREDICTION 

Our best model was proven to be dine utuning a pre trained DeepLab model with AI crowd submission of 0.907. It is located in the notebook experiments_deeplab.ipynb. 

## REPORT
A 4 page scientific report describes the most relevant feature engineering techniques and implementations that we worked on, explains how and why these techniques improved our predictions and includes an ethical consideration associated to this machine learining problem. 

## RUN.PY
To run the run.py and obtain our best prediction the user would need to have the following necessary libraries, as our model is too heavy to upload to github so we made an accessible link through google drive and necessitates this library and 160MB to store the model:
import gdown 

import torch
import torchvision
from torchvision import models
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

from data import mask_to_submission
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import os
import seaborn as sns
import pandas as pd

import os
import numpy as np
import matplotlib.image as mpimg
import re
