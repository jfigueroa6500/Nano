import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import helper.ImageProcessing
import matplotlib.pyplot as plt
import numpy as np


def create_model(arch, hidden_units):
    
    print("designing model....")

    #Loading Pretrained Model
    if arch.lower() == "alexnet":
        model = models.alexnet(pretrained=True)
        in_feat = 2208
    elif arch.lower() == "vgg16":
        model = models.vgg16(pretrained=True)
        in_feat = 25088
    else:
        # We dont support the entered model architecture so return to start over
        print("Model architecture: {}, is not supported. \n Please try alexnet or vgg16".format(arch.lower()))
        return 0
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Create our classifier to replace the current one in the model
    model.classifier = nn.Sequential(nn.Linear(in_feat,hidden_units),
                                     nn.ReLU(),         
                                     nn.Dropout(0.5),                            
                                     nn.Linear(hidden_units,102),
                                     nn.LogSoftmax(dim=1))

    print("modeling process completed!")
    return model

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, use_gpu):
   
    print("Training the model...\n")

    # Use the GPU if its available
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set the model to the device for training
    model.to(device)
    
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 25
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
        
            images, labels = images.to(device), labels.to(device)  
        
            optimizer.zero_grad()  
            
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"validation loss: {test_loss/len(validloader):.3f}.. "
                      f"validation accuracy: {accuracy/len(validloader):.3f}")
            
                running_loss = 0
                model.train()
                    

def save_model(model, train_data, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch):
   
    print("Saving the model...")

    # Save the train image dataset
    model.class_to_idx = train_datasets.class_to_idx

    if arch.lower() == "vgg19":
        input_features = 25088
    elif arch.lower() == "densenet161":
        input_features = 2208

    # Save other hyperparamters
    checkpoint = {'input_size': input_features,
                'output_size': 102,
                'hidden_units': hidden_units,
                'arch': arch,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'classifier' : model.classifier,
                'epochs': epochs,
                'criterion': criterion,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}


    torch.save(checkpoint, 'checkpoint.pth')
    print("Done saving the model")
    
def load_model(checkpoint_file):
    
    print("Loading the model...")
    
   
    checkpoint = torch.load(checkpoint_file)
   
    if(checkpoint['arch'].lower() == 'vgg19' or checkpoint['arch'].lower() == 'densenet161'):
        model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)


    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    print("Done loading the model")
    return model    

