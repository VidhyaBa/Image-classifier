#ACKNOWLEDGEMENT: I referred the following website and github repositories respectively for debugging/help:
#https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
#https://github.com/sedemmler/AIPND_image_classification/blob/master/predict.py
# Imports python modules
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import json
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

# Main program function defined below
def main():
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    # set directories
    image_path = in_arg.path_to_image
    model_reloaded = load_checkpoint(in_arg.checkpoint)
    gpu = in_arg.gpu
    with open(in_arg.category_names, 'r') as f:
        category_to_name = json.load(f)
    probs, classes, flowers = predict(image_path, model_reloaded,category_to_name,gpu,in_arg.top_k)
    print('Flower and probability:{}, {}'.format(flowers,probs))
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    criterion = checkpoint['criterion']
    # model architecture
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)    
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Architecture not recognized")

    model.class_to_idx = checkpoint['mapping_of_classes']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    im = Image.open(image_path)
    #check if width > height
    print('checking size')
    if im.size[0] > im.size[1]:
        ratio = im.size[1]/256
        new_im_size_0 = im.size[0]/ratio
        im.thumbnail((new_im_size_0, 256)) #constrain the height to be 256
    else:
        ratio = im.size[0]/256
        new_im_size_1 = im.size[1]/ratio
        im.thumbnail((256, new_im_size_1)) #otherwise constrain the width
        
    im_l_coord = (im.width-224)/2
    im_u_coord = (im.height-224)/2
    im_r_coord = im_l_coord + 224
    im_b_coord = im_u_coord + 224   
    img_cropped = im.crop((im_l_coord, im_u_coord, im_r_coord,im_b_coord))
    scaled_image = np.array(img_cropped)/255
    averg = np.array([0.485, 0.456, 0.406]) # mean for normalization
    std = np.array([0.229, 0.224, 0.225]) # std for normalization
    norm_image = (scaled_image - averg)/std
    processed_image = norm_image.transpose((2, 0, 1))
    print('processed image')
    return processed_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    if title:
        plt.title(title)

    # matplotlib assumes color as third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, cat_to_name, gpu, num_of_values):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img=process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    model.eval()
    if gpu:
        model.cuda()
        img = img.cuda()
    img.unsqueeze_(0)

# Predict first 5 prob,class
    probs = torch.exp(model(img)) 
    first_probs, first_labels = probs.topk(num_of_values) 
    first_probs = first_probs.cpu()
    first_labels = first_labels.cpu()
    first_probs = first_probs.detach().numpy().tolist()[0] 
    first_labels = first_labels.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_k_labels = [index_to_class[label] for label in first_labels]
    top_k_flowers = [cat_to_name[index_to_class[label]] for label in first_labels]
    
    return first_probs, top_k_labels, top_k_flowers  

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates command line arguments for options

    parser.add_argument('path_to_image', type=str, default='flowers/test/1/image_06743.jpg', help='path to image')

    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='specify checkpoint to load')

    parser.add_argument('--top_k', type=int, default=5, help='number of top classes to return')

    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of category to names')
    
    parser.add_argument('--gpu', type=bool, default=True, help='activate GPU mode')         
    

    # returns parsed argument collection
    return parser.parse_args()

# Call to main function to run the program
if __name__ == "__main__":
    main()
