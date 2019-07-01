#Defines the Data loading function for Predict.py
import torch
import torchvision
from torchvision import datasets, transforms, models
import argparse


def image_data(data_directory, batch_size):
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    
    #Training Image set Transforms
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                            transforms.RandomRotation(50),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    
    #Valid Image set Transforms
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    #Test Image set Transforms
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    #Loading Images
    train_data = datasets.ImageFolder(data_directory + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_directory + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_directory + '/valid', transform=valid_transforms)

    # defining the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle =True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle =True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle =True)
    
    image_datasets = [train_data, valid_data, test_data]
    dataloader = [trainloader, validloader, testloader]
    
    return trainloader, validloader, testloader, train_data


