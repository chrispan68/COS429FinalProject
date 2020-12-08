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

sys.path += ['data']
from load_data import load_data

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False, dir=None):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy {:.4}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    state = {'state_dict': best_model_wts,
             'optimizer': optimizer.state_dict()}
    torch.save(state, dir)
    return model, val_acc_history, train_acc_history, val_loss_history, train_loss_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True   

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, (input_size, input_size)

def load_checkpoint(model, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint at '{}'".format(filename))
    else:
        print("No checkpoint found at '{}'".format(filename))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="Location of the fine-tuning date. Follow ImageFolder structure")
    parser.add_argument("--model_name", type=str, help="Type of model architecture to fine tune",choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"], default="squeezenet")
    parser.add_argument("--num_classes", type=int, help="Number of classes in output", default=2)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs", default=20)
    parser.add_argument("--batch_size", type=int, help="Amount of training data per batch", default=32)
    parser.add_argument("--train_mode", type=str, help="Mode of training", choices=["scratch", "finetune", "feature_extract", "saved"])
    parser.add_argument("--lr", type=float, help="Learning rate of optimization", default=0.001)
    args = parser.parse_args()

    data_dir = args.data_dir
    model_name = args.model_name
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_mode = args.train_mode

    #Initialize the model and optimizer
    model_ft, input_size = initialize_model(model_name, num_classes, (train_mode == "feature_extract"), use_pretrained=(train_mode != "scratch")) 
    params_to_update = []
    print("Params to learn:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    optimizer_ft = optim.Adam(params_to_update, lr=args.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=0)

    #Load saved states
    if train_mode == "saved":
        #Load model
        load_checkpoint(model_ft, os.path.join(data_dir, "checkpoint.pth.tar"))
        set_parameter_requires_grad(model_ft, False)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: load_data(os.path.join(data_dir, x), input_size) for x in ['train', 'val', 'test']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataloaders_dict['test'] = torch.utils.data.DataLoader(image_datasets['test'])

    # Detect if we have a GPU available
    device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, val_acc, train_acc, val_loss, train_loss = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs, is_inception=(model_name=="inception"), dir=os.path.join(data_dir, "checkpoint.pth.tar"))
    
    plt.title('Loss Curve (Train Loss in Blue, Valid Loss in Red)')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    train_plt = plt.plot(range(0, len(train_loss)), train_loss, 'b')
    valid_plt = plt.plot(range(0, len(val_loss)), val_loss, 'r')
    plt.show()

    plt.title('Accuracy Curve (Train Acc in Blue, Valid Acc in Red)')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    train_plt = plt.plot(range(0, len(train_acc)), train_acc, 'b')
    valid_plt = plt.plot(range(0, len(val_acc)), val_acc, 'r')
    plt.show()


if __name__ == "__main__":
    main()