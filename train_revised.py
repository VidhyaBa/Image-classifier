#ACKNOWLEDGEMENT: I referred the following website and github repositories respectively for debugging/help:
#https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
#https://github.com/sedemmler/AIPND_image_classification/blob/master/train.py
# Imports python modules
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


# Main program function defined below
def main():
     # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
     # set directories
    data_dir = in_arg.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Defined transforms for the training, validation, and testing sets
 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    if in_arg.gpu is True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    #set architecture for model and derive input size
    if in_arg.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    elif in_arg.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif in_arg.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    else:
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_features
        
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Hyperparameters for our network
    hidden_sizes = in_arg.hidden_units
    output_size = 102
   
    #create classifier
    classifier = OrderedDict()
    for i in range(len(hidden_sizes)+1):
        if i==0:
            classifier.update({'fc{}'.format(i+1):nn.Linear(input_size,hidden_sizes[i]),
            'relu{}'.format(i+1):nn.ReLU(),'dropout{}'.format(i+1):nn.Dropout(0.2)})
        elif i==(len(hidden_sizes)):
            classifier.update({'fc{}'.format(i+1):nn.Linear(hidden_sizes[i-1],output_size),'output':nn.LogSoftmax(dim=1)})
        else:
            classifier.update({'fc{}'.format(i+1):nn.Linear(hidden_sizes[i-1],hidden_sizes[i]),
                               'relu{}'.format(i+1):nn.ReLU(),'dropout{}'.format(i+1):nn.Dropout(0.2)})

    
    model.classifier = nn.Sequential(classifier)
    print(model)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
     
    model.to(device)
    #  Train the network here
    epochs = in_arg.epochs
    print_every = 40
    steps = 0


    for e in range(epochs):
        training_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),"Training Loss: {:.4f}".format(training_loss/print_every))
                training_loss = 0   
        
                valid_correct = 0
                valid_total = 0
                valid_loss = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model.forward(images)
                        _, predicted = torch.max(outputs.data, 1)
                        valid_loss += criterion(outputs, labels).item()
                        valid_total += labels.size(0)
                        valid_correct += (predicted == labels).sum().item()
                    print('Epoch: {}/{}... '.format(e+1, epochs),'Validation Loss: {:.3f}..'.format(valid_loss / len(validloader)))
                    print('Accuracy of the network on the validation images: %d %%' % (100 * valid_correct / valid_total))
         
    
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': input_size,
             'output_size': output_size,
             'epochs': epochs,
             'optimizer' : optimizer.state_dict,
             'criterion' : criterion.state_dict,
             'arch' : in_arg.arch,     
             'mapping_of_classes': model.class_to_idx,
             'classifier': model.classifier,
             'state_dict': model.state_dict()}

    torch.save(checkpoint,'checkpoint.pth')
    
    
    # Functions defined below
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

    parser.add_argument('data_directory', type=str, default='flowers', help='path to folder of images')

    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save checkpoints in given directory')

    parser.add_argument('--arch', type=str, default='vgg19', choices=['vgg13', 'vgg16', 'densenet121'], help='select network architecture')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='set the learning rate')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[4096,2048], help='set number of hidden units in layers')
    parser.add_argument('--epochs', type=int, default=4, help='set number of epochs for training phase')  
    parser.add_argument('--gpu', type=bool, default=True, help='activate GPU mode')         
    

    # returns parsed argument collection
    return parser.parse_args()

# Call to main function to run the program
if __name__ == "__main__":
    main()