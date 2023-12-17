
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
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union

from scripts.evaluation import metrics, DiceLoss, FocalLoss, metrics_torch, BinaryFocalLoss, FraudLoss
from scripts.models import VGG13, UNET

from scripts.preprocessing import *

def train_model(X, y, config, use_model=None):

    #road seg dataset
    #X, y = road.x, road.y
    if (config['test_size'] != 0):
      X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=config['test_size'], shuffle=True) #Split data
      validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32), torch.tensor(y_validation, dtype=torch.float32))
      validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size']) #create
    else:
      X_train, y_train = (X.copy(), y.copy())

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=torch.cuda.is_available(),
                              drop_last = False, num_workers = 2) #create train data loader

    n_epochs = config['n_epochs'] #Training Loop Start

    device = config['device']
    model_dict = {"VGG16": VGG13, "U-Net": UNET}
    optimizer_dict = {"AdamW": optim.AdamW}

    loss_function_dict = {"CrossEntropy": nn.CrossEntropyLoss, "BCELoss": nn.BCELoss, "Dice": DiceLoss,
     "Focal": FocalLoss, "binaryFocal": BinaryFocalLoss, "FraudLoss": FraudLoss}

    scheduler_dict  = {'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingLR}
    scheduler_kwargs_dict = {'CosineAnnealing': dict(T_max = n_epochs*len(train_loader.dataset))}

    prediction_transform_dict = {"PatchPred": PatchPred, "identity": identity_func, "identity_detach": identity_detach,
     "numpy_and_binarize": numpy_and_binarize, "binarize": binarize}
    labels_transform_dict = {"unet_reshape": permutate_labels, "identity": identity_func}
    
    if (use_model is not None):
      model = use_model
    else: 
      model = model_dict[config['model']]().to(device=device) #Get model
    
    optimizer = optimizer_dict[config['optimizer']](model.parameters(),lr = config['learning_rate']) #Get optimizer
    if ("weight_decay" in list(config.keys())): 
      optimizer = optimizer_dict[config['optimizer']](model.parameters(),lr = config['learning_rate'], weight_decay=config['weight_decay']) #Get optimizer
    criterion = loss_function_dict[config['loss_function']]() #Get loss function

    scheduler = scheduler_dict[config['scheduler']](optimizer, **scheduler_kwargs_dict[config['scheduler']])

    prediction_transform = prediction_transform_dict[config['prediction_transform']]
    labels_transform = labels_transform_dict[config['labels_transform']]


    print("X_train shape:", X_train.shape, y_train.shape)
    metrics_dict = {"training": [], "validation": []}
    for epoch in range(n_epochs):

        #Model training
        model.train()
        train_metrics = {'Accuracy': 0, 'F1-score': 0}
        total_train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = (inputs.to(device=device), labels.to(device=device))

            optimizer.zero_grad()
            inputs = torch.permute(inputs, (0, 3, 2, 1)) #batch_size, channels, width, height
            prediction = model(inputs)

            labels = labels_transform(labels)
            loss = criterion(prediction, labels)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            prediction_class = prediction_transform(prediction)
            #labels = np.array(labels.detach().cpu().numpy())
            #batch_metrics = metrics(prediction_class.flatten(), labels.flatten())
            batch_metrics = metrics_torch(prediction_class, labels)
            train_metrics['Accuracy'] += batch_metrics['Accuracy']
            train_metrics['F1-score'] += batch_metrics['F1-score']

        train_metrics['Accuracy'] /= len(train_loader)
        train_metrics['F1-score'] /= len(train_loader)
        average_train_loss = total_train_loss/len(train_loader)

        if (epoch + 1) % config['validate_every'] == 0:
            model.eval()
            validation_metrics = {'Accuracy': 0, 'F1-score': 0}
            total_validation_loss = 0
            average_validation_loss = 0
            if (config['test_size'] != 0):
              for validation_inputs, validation_labels in validation_loader:
                  validation_inputs, validation_labels = (validation_inputs.to(device=device), validation_labels.to(device=device))
                  with torch.no_grad():

                      validation_inputs = torch.permute(validation_inputs, (0, 3, 2, 1))
                      prediction = model(validation_inputs)

                      validation_labels = labels_transform(validation_labels)

                      validation_loss = criterion(prediction, validation_labels)
                      total_validation_loss += validation_loss.item()

                      prediction_class = prediction_transform(prediction)
                      #validation_labels = np.array(validation_labels.detach().cpu().numpy())
                      #batch_metrics = metrics(prediction_class.flatten(), validation_labels.flatten())
                      batch_metrics = metrics_torch(prediction_class, validation_labels)
                      validation_metrics['Accuracy'] += batch_metrics['Accuracy']
                      validation_metrics['F1-score'] += batch_metrics['F1-score']

              validation_metrics['Accuracy'] /= len(validation_loader)
              validation_metrics['F1-score'] /= len(validation_loader)
              average_validation_loss = total_validation_loss/len(validation_loader)

              metrics_dict["training"].append(train_metrics)
              metrics_dict["validation"].append(validation_metrics)
              
            print('Epoch:', '%03d' % (epoch + 1), 'train loss =', '{:.6f}'.format(average_train_loss),
                   'val loss =', '{:.6f}'.format(average_validation_loss),'train accuracy =','{:.4f}'.format(train_metrics['Accuracy']),
                     'val accuracy =','{:.4f}'.format(validation_metrics['Accuracy']), 'validation F1', '{:.4f}'.format(validation_metrics['F1-score']))

    return model, metrics_dict