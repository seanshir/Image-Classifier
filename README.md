# Developing an Image Classifier with Deep Learning

In this project, first I developed a code for an image classifier built with PyTorch, then convert it into a command line application.


## Part 1 - Development

The project development is where I created and trained a neural network. 
A summary of steps taken are as follows:

- Load a pre-trained network
- Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
- Train the classifier layers using backpropagation using the pre-trained network to get the features
- Track the loss and accuracy on the validation set to determine the best hyperparameters

## Part 2 - Building the command line application
The application part of the project is taking what I made during development phase and convert it to a command line and argument based application that others can use. 
The application includes a pair of Python scripts that run from the command line. 
This is implemented as a generalization of development phase (part 1). User of the command line app can load in any torchvision network, and specify the desired hidden units and output. 


## File Descriptions
- Image Classifier Project.ipynb: python notebook with work done for Development part 
- Image Classifier Project.html: html version of python notebook 
- train.py: Trains a new network on a dataset and save the model as a checkpoint.  
- predict.py: Uses a trained network to predict the class for an input image.   
- helper.py: Includes the functions and classes relating to the model. 
- utility.py: Includes the functions like loading data and preprocessing images. 

## Train a new network on a data set with train.py
### Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

#### Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg16"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu

## Predict flower name from an image with predict.py along with the probability of that name.
That is, pass in a single image /path/to/image and return the flower name and class probability.

### Basic usage: python predict.py /path/to/image checkpoint
#### Options:
- Return top KK most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu
