# Image-classifier
Implementation of neural network using PyTorch for image classification
This directory consists of three files:
1. Jupyter notebook consisting of the network model selection,training and prediction code implemented as an HTML file

2. train.py - Python code that carries out the training and validation of the network model:
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training: python train.py data_dir --gpu

3. predict.py - Python code that predicts the flower category and probability based on the trained model
Predict flower name from an image with predict.py along with the probability of that name. Pass in a single image /path/to/image and return the flower name and class probability.

    Basic usage: python predict.py /path/to/image checkpoint
    Options:
        Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
        Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
        Use GPU for inference: python predict.py input checkpoint --gpu



