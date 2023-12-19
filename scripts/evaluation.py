############################################################
#
#This script is designed to contain functions used to evaluate the performance of our models
#
############################################################

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scripts.preprocessing import *

import torch
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union
import torch.nn as nn


def metrics_torch(y_pred, y, significant_digits = 5):
    """"
    Function evaluates models for binary classification tasks
    Inputs are 2 1D torch.tensors
    y_pred is your predicted numpy array values are {0, 1}
    y is the actual labels/ground truth values, also in {0, 1}
    """
    y_pred, y = torch.flatten(y_pred), torch.flatten(y)
    TP = torch.sum(y_pred*y)
    TN = torch.sum(y_pred + y == 0)
    FP = torch.sum(y_pred) - TP
    FN = torch.sum(y) - TP

    if (TP+FP != 0):
        P = TP/(TP+FP)
    else:
        P = 0

    if (TP+FN != 0):
        R = TP/(TP+FN)
    else:
        R = 0

    if (P+R != 0):
        F1 = 2*(P*R)/(P+R)
    else:
        F1 = 0
    metric_formulas = {"Accuracy": (TP + TN)/(TP + TN + FP + FN), "Precision": P,
                        "Recall": R, "F1-score":F1}
    return metric_formulas

  
def metrics(y_pred, y, significant_digits = 5):
    """"
    Function evaluates models for binary classification tasks
    Inputs are 2 1D numpy arrays
    y_pred is your predicted numpy array values are {0, 1}
    y is the actual labels/ground truth values, also in {0, 1}
    """
    TP, TN, FP, FN = (0 for i in range(4))
    TP = np.sum(y_pred + y == 2)
    TN = sum(y_pred + y == 0)
    FP = sum(y_pred) - TP
    FN = sum(y) - TP

    if (TP+FP != 0):
        P = TP/(TP+FP)
    else:
        P = 0

    if (TP+FN != 0):
        R = TP/(TP+FN)
    else:
        R = 0

    if (P+R != 0):
        F1 = 2*(P*R)/(P+R)
    else:
        F1 = 0
    metric_formulas = {"Accuracy": (TP + TN)/(TP + TN + FP + FN), "Precision": P,
                        "Recall": R, "F1-score":F1}
    return metric_formulas

def visualize_predicition(img, gt, scaler, model, n_patches):
    """
    Takes model and scaler and runs pipeline on an image and plots it versus the groundtruth
    """
    output_shape = gt.shape

    patches = img_to_patches(img, number_of_patches=n_patches)
    features = scaler.transform(np.array([patch_to_features(patch) for patch in patches]))

    predicition_patches = model.predict(features)
    img_mask = predicted_patch_labels_to_mask(predicition_patches, output_shape)

    fig, ax = plt.subplots(1, 2, figsize = (4, 8))
    ax[0].imshow(gt)
    ax[1].imshow(img_mask)
    fig.suptitle("Visualization and metrics for 1 mask", y = 0.63)
    plt.show()

    #Get image metrics
    gt_patches = img_to_patches(gt, n_patches)
    labels_patches = np.array([groundtruth_patch_to_label(patch,) for patch in gt_patches])

    return metrics(predicition_patches, labels_patches)

def mask_predictions(scaler, model, n_patches, img_directory, gt_directory, dataset_size=100):

    for root, dirs, files in sorted(os.walk(img_directory)): image_filenames = sorted(files)

    AICROWD_PATCHSIZE = 16
    avg_metrics = {'Accuracy': 0, 'F1-score': 0}
    for file in image_filenames[:dataset_size]:

        img = cv.imread(img_directory+'/'+file)
        gt = cv.imread(gt_directory+'/'+file)

        #Patch the input image and generate predictions for patches
        img_patches = img_to_patches(img, number_of_patches=n_patches)
        img_features = scaler.transform(np.array([patch_to_features(patch) for patch in img_patches]))
        predicition_img_patches = model.predict(img_features)

        #reconstruct from predicted patch labels a mask
        predicted_mask = predicted_patch_labels_to_mask(predicition_img_patches, output_shape=img.shape)

        #get groundtruth patches and mask like AICrowd submissions
        n_patches_gt = int((gt.shape[0]/AICROWD_PATCHSIZE)**2)
        gt_patches = img_to_patches(gt, number_of_patches=n_patches_gt)
        labels_patches = np.array([groundtruth_patch_to_label(patch, foreground_threshold=.25) for patch in gt_patches])
        gt_patched_mask = predicted_patch_labels_to_mask(labels_patches, gt.shape)

        file_metrics = metrics(predicted_mask.flatten(), gt_patched_mask.flatten())
        avg_metrics['Accuracy'] += file_metrics['Accuracy']
        avg_metrics['F1-score'] += file_metrics['F1-score']

    avg_metrics['Accuracy'] /= len(image_filenames[:dataset_size])
    avg_metrics['F1-score'] /= len(image_filenames[:dataset_size])
    return avg_metrics

def mask_predictions_vgg(scaler, model, n_patches, img_directory, gt_directory, dataset_size=100):


    for root, dirs, files in sorted(os.walk(img_directory)): image_filenames = sorted(files)

    AICROWD_PATCHSIZE = 16
    avg_metrics = {'Accuracy': 0, 'F1-score': 0}
    for file in image_filenames[:dataset_size]:

        img = cv.imread(img_directory+'/'+file)
        gt = cv.imread(gt_directory+'/'+file)

        #Patch the input image and generate predictions for patches
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        img_patches = torch.permute(torch.tensor(np.array(img_to_patches(img, number_of_patches=n_patches)),  dtype=torch.float32), (0, 3, 2, 1)).to(device=DEVICE)
        prediction_img_patches = model(img_patches)
        prediction_class_img_patches = PatchPred(prediction_img_patches)

        #reconstruct from predicted patch labels a mask
        predicted_mask = predicted_patch_labels_to_mask(prediction_class_img_patches, output_shape=img.shape)

        #get groundtruth patches and mask like AICrowd submissions
        n_patches_gt = int((gt.shape[0]/AICROWD_PATCHSIZE)**2)
        gt_patches = img_to_patches(gt, number_of_patches=n_patches_gt)
        labels_patches = np.array([groundtruth_patch_to_label(patch, foreground_threshold=.25) for patch in gt_patches])
        gt_patched_mask = predicted_patch_labels_to_mask(labels_patches, gt.shape)

        file_metrics = metrics(predicted_mask.flatten(), gt_patched_mask.flatten())
        avg_metrics['Accuracy'] += file_metrics['Accuracy']
        avg_metrics['F1-score'] += file_metrics['F1-score']

    avg_metrics['Accuracy'] /= len(image_filenames[:dataset_size])
    avg_metrics['F1-score'] /= len(image_filenames[:dataset_size])
    return avg_metrics

def visualize_predicition_vgg(img, gt, scaler, model, n_patches):
    """
    Takes model and scaler and runs pipeline on an image and plots it versus the groundtruth
    """
    output_shape = gt.shape

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    img_patches = torch.permute(torch.tensor(np.array(img_to_patches(img, number_of_patches=n_patches)),  dtype=torch.float32), (0, 3, 2, 1)).to(device=DEVICE)
    prediction_img_patches = model(img_patches)
    prediction_class_img_patches = PatchPred(prediction_img_patches)
    img_mask = predicted_patch_labels_to_mask(prediction_class_img_patches, output_shape)

    AICROWD_PATCHSIZE = 16
    n_patches_gt = int((gt.shape[0]/AICROWD_PATCHSIZE)**2)
    gt_patches = img_to_patches(gt, number_of_patches=n_patches_gt)
    labels_patches = np.array([groundtruth_patch_to_label(patch, foreground_threshold=.25) for patch in gt_patches])
    gt_patched_mask = predicted_patch_labels_to_mask(labels_patches, gt.shape)

    fig, ax = plt.subplots(1, 2, figsize = (4, 8))
    ax[0].imshow(gt_patched_mask)
    ax[1].imshow(img_mask)
    fig.suptitle("Visualization and metrics for 1 mask", y = 0.63)
    plt.tight_layout()
    plt.show()

    return metrics(img_mask.flatten(), gt_patched_mask.flatten())

def mask_predictions_unet(model, img_directory, gt_directory, dataset_size=100):

    for root, dirs, files in sorted(os.walk(img_directory)): image_filenames = sorted(files)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AICROWD_PATCHSIZE = 16
    avg_metrics = {'Accuracy': 0, 'F1-score': 0}
    for file in image_filenames[:dataset_size]:

        img = cv.imread(img_directory+'/'+file)
        gt = parse_mask_to_binary(cv.imread(gt_directory+'/'+file))

        img = torch.tensor(img[:,:,:,None].T, dtype=torch.float32).to(device=DEVICE)
        predicted_mask = np.where(identity_detach(model(img)) > 0.5, 1, 0)

        file_metrics = metrics(predicted_mask.T.flatten(), gt.flatten())
        avg_metrics['Accuracy'] += file_metrics['Accuracy']
        avg_metrics['F1-score'] += file_metrics['F1-score']

    avg_metrics['Accuracy'] /= len(image_filenames[:dataset_size])
    avg_metrics['F1-score'] /= len(image_filenames[:dataset_size])
    return avg_metrics

def visualize_predicition_unet(img, gt, model):
    """
    Takes model and scaler and runs pipeline on an image and plots it versus the groundtruth
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    img = torch.tensor(img[:,:,:,None].T, dtype=torch.float32).to(device=DEVICE)
    img_mask = np.where(identity_detach(model(img)) > 0.5, 1, 0)

    fig, ax = plt.subplots(1, 2, figsize = (4, 8))
    ax[0].imshow(gt)
    ax[1].imshow(img_mask[0][0].T)
    fig.suptitle("Visualization and metrics for 1 mask", y = 0.63)
    plt.show()

    #Get image metrics

    return metrics(img_mask.T.flatten(), gt.flatten())


class DiceLoss(nn.Module):
  def __init__(self, smooth=1):
      super(DiceLoss, self).__init__()
      self.smooth = smooth # adding small numer to avoid division by zero

  def forward(self, prediction, target):
      intersection = (prediction * target).sum()
      union = prediction.sum() + target.sum() + 1
      dice = (2 * intersection + 1) / union
      return 1 - dice

class FocalLoss(nn.Module):
  """Computes the focal loss between input and target
  as described here https://arxiv.org/abs/1708.02002v2

  Args:
      gamma (float):  The focal loss focusing parameter.
      weights (Union[None, Tensor]): Rescaling weight given to each class.
      If given, has to be a Tensor of size C. optional.
      reduction (str): Specifies the reduction to apply to the output.
      it should be one of the following 'none', 'mean', or 'sum'.
      default 'mean'.
      ignore_index (int): Specifies a target value that is ignored and
      does not contribute to the input gradient. optional.
      eps (float): smoothing to prevent log from returning inf.
  """
  def __init__(
          self,
          gamma=2,
          weights: Union[None, Tensor] = None,
          reduction: str = 'mean',
          ignore_index=-100,
          eps=1e-16
          ) -> None:
      super().__init__()
      if reduction not in ['mean', 'none', 'sum']:
          raise NotImplementedError(
              'Reduction {} not implemented.'.format(reduction)
              )
      assert weights is None or isinstance(weights, Tensor), \
          'weights should be of type Tensor or None, but {} given'.format(
              type(weights))
      self.reduction = reduction
      self.gamma = gamma
      self.ignore_index = ignore_index
      self.eps = eps
      self.weights = weights

  def _get_weights(self, target: Tensor) -> Tensor:
      if self.weights is None:
          return torch.ones(target.shape[0])
      weights = target * self.weights
      return weights.sum(dim=-1)

  def _process_target(
          self, target: Tensor, num_classes: int, mask: Tensor
          ) -> Tensor:
      
      #convert all ignore_index elements to zero to avoid error in one_hot
      #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
      target = target * (target!=self.ignore_index) 
      target = target.reshape(-1)
      return one_hot(target, num_classes=num_classes)

  def _process_preds(self, x: Tensor) -> Tensor:
      if x.dim() == 1:
          x = torch.vstack([1 - x, x])
          x = x.permute(1, 0)
          return x
      return x.view(-1, x.shape[-1])

  def _calc_pt(
          self, target: Tensor, x: Tensor, mask: Tensor
          ) -> Tensor:
      p = target * x
      p = p.sum(dim=-1)
      p = p * ~mask
      return p

  def forward(self, x: Tensor, target: Tensor) -> Tensor:
      assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
          'The predictions values should be between 0 and 1, \
              make sure to pass the values to sigmoid for binary \
              classification or softmax for multi-class classification'
      )
      mask = target == self.ignore_index
      mask = mask.reshape(-1)
      x = self._process_preds(x)
      num_classes = x.shape[-1]
      target = self._process_target(target, num_classes, mask)
      weights = self._get_weights(target).to(x.device)
      pt = self._calc_pt(target, x, mask)
      focal = 1 - pt
      nll = -torch.log(self.eps + pt)
      nll = nll.masked_fill(mask, 0)
      loss = weights * (focal ** self.gamma) * nll
      return self._reduce(loss, mask, weights)

  def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
      if self.reduction == 'mean':
          return x.sum() / (~mask * weights).sum()
      elif self.reduction == 'sum':
          return x.sum()
      else:
          return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure the inputs are squeezed to remove the channel dimension if it's 1
        inputs = inputs.squeeze(1)
        targets = targets.squeeze(1)

        # Calculate the binary cross entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate the focal loss components
        targets = targets.type(inputs.type())  # Ensure same type
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class FraudLoss(nn.Module):
    def __init__(self):
      super(FraudLoss, self).__init__()
      self.binaryCE = nn.BCELoss()
      self.dice = DiceLoss()

    def forward(self, prediction, target):
      return self.binaryCE(prediction, target) + self.dice(prediction, target)

# Example usage
# pred = torch.randn(4, 1, 256, 256)  # Example prediction tensor
# target = torch.randint(0, 2, (4, 1, 256, 256))  # Example target tensor
# criterion = BinaryFocalLoss()
# loss = criterion(pred, target)
DEVICE = "mps" if getattr(torch, "has_mps", False) else "cuda" if torch.cuda.is_available() else "cpu"

def test_masks(test_directory, model, visualize_number=6, deep=False):

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
      print(test_img.shape)
      mask = np.where(identity_detach(model(test_img)['out'])>0.5, 1, 0)[0][0]

    if (fdx%5 == 0): print(f"data/test_masks/{file.split('/')[-1]}")
    output_filename = f"data/test_masks/{file.split('/')[-1]}"
    output_masks_filenames.append(output_filename)
    cv.imwrite(output_filename, 255*mask.T)

    output_masks.append(mask.T)

  test_ids =  np.random.choice(len(output_masks), size = visualize_number)

  for i in range(visualize_number):
    plt.figure(figsize = (3, 3))
    test_img = cv.imread(test_filepaths[test_ids[i]])
    out_img = color.gray2rgb(255*output_masks[test_ids[i]])
    #print(test_img, out_img)
    cat_img = np.concatenate([test_img, out_img], axis=1)
    plt.imshow(cat_img)
    plt.show()
  return output_masks, output_masks_filenames