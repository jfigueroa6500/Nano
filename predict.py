import nnModel
import torchvision
from torchvision import datasets, transforms, models
import helper.JsonFun
import helper.Dataloader
import argparse



#Argument options
parser = argparse.ArgumentParser(description="Load a Network to use ")

parser.add_argument('--gpu', default=False, action='store_true',
                    help="GPU option")
parser.add_argument('--data_directory', default="/flowers")
parser.add_argument('--checkpoint',
                    help="Load pth file.")
parser.add_argument('--top_k', default=3, type=int,
                    help="The amount of likley predictions")
parser.add_argument('--category_names', default = './cat_to_name.json',
                    help="load category Labels")

#Loading Arguments
args = parser.parse_args()
use_gpu = args.gpu
data_directory = args.data_directory
checkpoint = args.checkpoint
top_k = args.top_k
category_name = args.category_names



#Loading the Trained Model
model = nnModel.load_model(checkpoint)

#Load JSON FUNCTION from Helper folder
categories = helper.JsonFun.load_json(category_name)

#loading Predict function to return likely predictions
nnModel.predict(categories, data_directory, model, use_gpu, top_k)
