### INSERT TEST DIRECTORY HERE

test_directory = "data/test_set_images/"

### MAKE SURE YOU HAVE A "data/test_masks/" DIRECTORY FOR OUTPUT MASKS

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


DEVICE = "mps" if getattr(torch, "has_mps", False) else "cpu"
print(DEVICE)

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

# REQUIRES gdown

url = "https://drive.google.com/file/d/1K6U2lSOekpkg94-ZO_phWWSeB-zZzu1f/view?usp=sharing"
id = url.split('/')[-2]

prefix = 'https://drive.google.com/uc?/export=download&id='

gdown.download(prefix+id)

def identity_detach(x):
    return x.detach().cpu().numpy()
#PATH TO TEST MASKS
def test_masks(test_directory, model, visualize_number=6, deep=False, viz_all = False):

  test_filepaths = []
  for root, dirs, files in sorted(os.walk(test_directory))[1:]: test_filepaths.append(root+'/'+files[0])

  model.eval()

  output_masks = [] #Here we go baby
  output_masks_filenames = []
  for fdx, file in enumerate(sorted(test_filepaths)):

    test_img = cv.imread(file)
    test_img = torch.tensor(test_img[:,:,:,None].T, dtype=torch.float32).to(device=DEVICE)

    if (not deep):
      mask = np.where(identity_detach(model(test_img)) > 0.5, 1, 0)[0][0]
    else:
      #print(test_img.shape)
      mask = np.where(identity_detach(model(test_img)['out'])>0.5, 1, 0)[0][0]

    if (fdx%5 == 0): print(f"data/test_masks/{file.split('/')[-1]}")
    output_filename = f"data/test_masks/{file.split('/')[-1]}"
    output_masks_filenames.append(output_filename)
    cv.imwrite(output_filename, 255*mask.T)

    output_masks.append(mask.T)

  test_ids =  np.random.choice(len(output_masks), size = visualize_number)
  if (not viz_all):
    for i in range(visualize_number):
      plt.figure(figsize = (3, 3))
      test_img = cv.imread(test_filepaths[test_ids[i]])
      out_img = color.gray2rgb(255*output_masks[test_ids[i]])
      #print(test_img, out_img)
      cat_img = np.concatenate([test_img, out_img], axis=1)
      plt.imshow(cat_img)
      plt.show()
  if (viz_all):
    for i in range(17):
      fig, ax = plt.subplots(1, 3, figsize = (9, 3))
      num_im = 3
      if (i==16):
         num_im = 2
      for j in range(num_im):
        test_img = cv.imread(test_filepaths[i*3 + j])
        out_img = color.gray2rgb(255*output_masks[i*3+j])
        cat_img = np.concatenate([test_img, out_img], axis=1)
        ax[j].imshow(cat_img)
      plt.tight_layout()
      plt.show()

  return output_masks, output_masks_filenames

def create_deeplabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, outputchannels, kernel_size=(1,1), stride=(1,1))
    return model

deeplab = create_deeplabv3().to(DEVICE)

model = torch.load("deeplab_best.pth")
model.eval()
predicted_test_masks, mask_filenames = test_masks(test_directory, model = model,
                                                  visualize_number=3, deep=True, viz_all=False)

mask_to_submission.masks_to_submission("submission_deeplab_best.csv", *mask_filenames)
print("Done!")