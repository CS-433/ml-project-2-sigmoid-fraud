############################################################
#
#This script is designed to contain functions used to evaluate the performance of our models
#
############################################################

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scripts.preprocessing import *

def metrics(y_pred, y, significant_digits = 5):
    """"
    Function evaluates models for binary classification tasks
    Inputs are 2 1D numpy arrays
    y_pred is your predicted numpy array values are {0, 1}
    y is the actual labels/ground truth values, also in {0, 1}
    """
    TP, TN, FP, FN = (0 for i in range(4))
    TP = sum(y_pred + y > 1)
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
    metric_formulas = {"Accuracy": round((TP + TN)/(TP + TN + FP + FN), significant_digits), "Precision": round(P, significant_digits),
                        "Recall": round(R, significant_digits), "F1-score":round(F1, significant_digits)}
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