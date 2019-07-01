
#imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

#Defines Images procession function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #pre-processing image
    image = Image.open(image_path)
# Resize the images where shortest side is 256 pixels, keeping aspect ratio. 
    if image.width > image.height:
        
        factor = image.width/image.height
        image = image.resize(size=(int(round(factor*256,0)),256))
    else:
        factor = image.height/image.width
        image = image.resize(size=(256, int(round(factor*256,0))))
# Crop out the center 224x224 portion of the image.

    image = image.crop(box=((image.width/2)-112, (image.height/2)-112, (image.width/2)+112, (image.height/2)+112))

    # Convert to numpy array
    np_image = np.array(image)
    np_image = np_image/255
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std 
# Reorder dimension for PyTorch
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    
    def imshow(image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax