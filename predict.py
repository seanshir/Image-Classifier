import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import helper
import utility

parser = argparse.ArgumentParser()
parser.add_argument('input', action='store', help='path to image to be classified')
parser.add_argument('checkpoint', action='store', help='path to stored model')
parser.add_argument('--top_k', action='store', type=int, default=1, help='how many most probable classes to print out')
parser.add_argument('--category_names', action='store', help='file which maps classes to names')
parser.add_argument('--gpu', action='store_true', help='use gpu to infer classes')
args=parser.parse_args()

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: device = "cpu"

model = utility.load_model(args.checkpoint, args.gpu).eval()
#process image to pytensor using device    
img = utility.process_image(args.input).to(device) 
#see how the model classifies
outputs = model(img) 
#take exponent to undo log_softmax and get prediction of model
prob = torch.exp(outputs)
#and get top k probabilities
result = torch.topk(prob, args.top_k)    
#get topk from pytroch tensor to numpy
top_probs = result[0][0].cpu().detach().numpy() 
#get index of top5 probabilities
classes = result[1][0].cpu().numpy() 

if(args.category_names != None): 
    classes = utility.get_class(classes, args.checkpoint, args.category_names)
else:
    classes= utility.get_class(classes, args.checkpoint, None)

utility.show_classes(top_probs, classes, args.top_k)