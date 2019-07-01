import torch
from torch import nn as nn
from torch import optim as optim
import nnModel
import helper.Dataloader
import argparse

# argument parser & arguments
parser = argparse.ArgumentParser(description="Train a Neural Network using transfer learning")
# 1. Get the directory to the image files to train with
parser.add_argument('--data_directory', default='./flowers',
                    help="Traing images location")
# 2. Get the directory to the image files to train with
parser.add_argument('--save_dir', default='./',
                    help="Where do you want to save your network")
# 3. Choose the architecture
parser.add_argument('--arch', default="vgg16",
                    help="The architecture the neural network will use please choose from alexnet or vgg16")
# 4. Set the hyperparameters: Learning Rate, Hidden Units, Training Epochs, Training batch size
parser.add_argument('--learning_rate', type=float, default="0.001",
                    help="learning rate for the model")
parser.add_argument('--hidden_units', type=int, default=1000,
                    help="number of units in the hidden layer")
parser.add_argument('--epochs', type=int, default=1,
                    help=" amount training epochs traing your model")
parser.add_argument('--batch_size', type=int, default=64,
                    help="batch size for processing")
# 5. Choose the GPU for training
parser.add_argument('--gpu', default=False, action='store_true',
                    help="Use GPU for training. Default is False")

# Collect the arguments
args = parser.parse_args()
data_directory = args.data_directory
save_directory = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
batch_size = args.batch_size
gpu = args.gpu

# Get the image data from the files and create the data loaders
trainloader, validloader, testloader, train_data = helper.Dataloader.image_data(data_directory, batch_size)

# Create the model. Returns 0 if model cant be created
model = nnModel.create_model(arch, hidden_units)

# If we sucessfully create a model continue with the training
if model != 0:
    # Define the optimizer & loss
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    # Train the model 
    nnModel.train_model(model, trainloader, validloader, criterion, optimizer, epochs, gpu)

    # Save the model
    nnModel.save_model(model, train_data, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch)
