import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        #create hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:]) 
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
    
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x)) #apply relu to each hidden node
            x = self.dropout(x) #apply dropout
        x = self.output(x) #apply output weights
        return F.log_softmax(x, dim=1) #apply activation log softmax
    
def loss_accuracy(model, loader, criterion, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"

    model.eval()
    accuracy = 0
    loss=0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            prob = torch.exp(outputs) 
            
            #The highest probability gives prediction
            results = (labels.data == prob.max(1)[1]) 
            accuracy+=results.type_as(torch.FloatTensor()).mean() 
            loss+=criterion(outputs,labels)
            
             #Finally the avg over all the input images would give the accuracy and loss
    return accuracy/len(loader), loss/len(loader) 
    
#And the function to train the network
def train_network(model, trainloader, validloader, epochs, print_every, criterion, optimizer, scheduler, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    
    steps=0
    # change to cuda
    model.to(device)
    #training mode
    model.train() 

    for e in range(epochs):
        scheduler.step() 
        running_loss=0
        for ii, (inputs,labels) in enumerate(trainloader):
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device) 
            #zero out gradients so that one forward pass doesn't pick up previous forward's gradients            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                accuracy,valid_loss = loss_accuracy(model,validloader,criterion,gpu)
                           
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss),
                      "Validation Accuracy: {:.4f}".format(accuracy))
                running_loss = 0
            model.train()