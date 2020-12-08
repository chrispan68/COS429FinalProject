from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from train import load_checkpoint, initialize_model, set_parameter_requires_grad

sys.path += ['data']
from load_data import load_data

def eval_model(model, dataloader, criterion, device):
    with torch.no_grad():
        print("Test:")
        probs = []
        labels = []
        testing_loss = 0
        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, batch_labels)

            probs.append(outputs)
            labels.append(batch_labels.data)
            # statistics
            testing_loss += loss.item() * inputs.size(0)
        
        probs = torch.cat(probs).data.cpu().numpy()
        labels = torch.cat(labels).data.cpu().numpy()
        probs = (np.exp(probs) / np.sum(np.exp(probs), axis=1)[:, None])[:,1]
        preds = probs > 0.5
        rocauc = roc_auc_score(labels, probs)
        confusion = confusion_matrix(labels, preds)
        fpr, tpr, thresholds = roc_curve(labels, probs)
        test_loss = testing_loss / len(dataloader.dataset)
        test_acc = np.sum(preds == labels) / len(preds)

        print('Test Loss: {:4f}'.format(test_loss))
        print('Test Acc: {:4f}'.format(test_acc))

        print('Confusion Matrix')
        print('\n'.join([''.join(['{:7}'.format(item) for item in row]) 
        for row in confusion]))

        print('ROC AUC: {:4f}'.format(rocauc))
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.3f)' % rocauc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="Location of the fine-tuning date. Follow ImageFolder structure")
    parser.add_argument("--model_name", type=str, help="Type of model architecture to fine tune",choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"], default="squeezenet")
    parser.add_argument("--checkpoint_loc", type=str, help="Location of the checkpoint to be used")
    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_loc = args.checkpoint_loc
    model_name = args.model_name

    if model_name == "inception":
        input_size = (299, 299)
    else:
        input_size = (224, 224)

    model, input_size = initialize_model(model_name, 2, True, use_pretrained=False) 
    load_checkpoint(model, checkpoint_loc)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_dataset = load_data(os.path.join(data_dir, 'test'), input_size)

    # Create training and validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    eval_model(model, dataloader, criterion, device)

if __name__ == "__main__":
    main()