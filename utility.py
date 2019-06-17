import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import helper
import json

def preprocess_img(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)


    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data)
    testloader = torch.utils.data.DataLoader(test_data)

    return train_data, valid_data, test_data, trainloader, validloader, testloader

def save_checkpoint(model, train_data, optimizer, save_dir, arch):
    model_checkpoint = {
            'input_size': model.classifier.hidden_layers[0].in_features,
            'output_size': model.classifier.output.out_features,
            'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
            'state_dict': model.classifier.state_dict(),
            'class_idx': train_data.class_to_idx,
            'optim_idx': optimizer.state_dict,
            'dropout': model.classifier.dropout.p,
            'arch':arch
                   }
    if(save_dir == None): torch.save(model_checkpoint, 'checkpoint.pth')
    else:torch.save(model_checkpoint, save_dir+'checkpoint.pth')

def load_model(filepath, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    
    checkpoint = torch.load(filepath)
    model = torch_model(checkpoint['arch'])
    newModel = helper.NeuralNetwork(checkpoint['input_size'],
                     checkpoint['output_size'],
                     checkpoint['hidden_layers'], 
                     checkpoint['dropout'])
    newModel.load_state_dict(checkpoint['state_dict'])
    model.classifier = newModel
    model = model.float().to(device)
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img_loader = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)     
        
    img_tensor = torch.from_numpy(np.expand_dims(np_image, axis=0)).float()
    
    return img_tensor

def torch_model(arch):
     try: 
            model = getattr(models, arch)(pretrained=True)
            return model
     except AttributeError:
         print("%s is not valid torchvision model" % arch)
         raise SystemExit
     else:
        print("error loading model")
        raise SystemExit

def get_class(classes, checkpoint, category_names):
    class_to_idx = torch.load(checkpoint)['class_idx'] 
    # geta dict with mapping (class index, class 'name')
    idx_to_class = {idx: pic for pic, idx in class_to_idx.items()} 
    
    #take index number, change to class number, and then to flower name
    if(category_names != None): 
        names = []
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        for i in classes:
            #convert index of top5 to class number
            category = idx_to_class[i] 
            #convert category/class number to flower name
            name = cat_to_name[category] 
            names.append(name)
        return names
    
    else: # Take the index number and change it to real class number
        class_id = []
        for i in classes:
            class_id.append(idx_to_class[i])
        return class_id

def show_classes(probabilities, classes, top_k):
    print('--------Predictions for Image--------')
    i = 0
    while (i < top_k):
        print('%*s. Class: %*s. Pr= %.4f'%(7, i+1, 3, classes[i], probabilities[i]))
        i += 1
        
def get_input_size(model, arch):
    input_size = 0
    
    if('vgg' in arch): return model.classifier[0].in_features
    elif('densenet' in arch): return model.classifier.in_features
    elif('squeezenet' in arch): return model.classifier[1].in_channels
    elif(('resnet' in arch) or ('inception'in arch) ): return model.fc.in_features
    elif('alexnet' in arch): return model.classifier[1].in_features
        
    if(input_size == 0): raise Error    
    return input_size