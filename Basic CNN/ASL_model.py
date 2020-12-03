'''Contains the architecture for a model that classifies ASL gestures as alphabets. Works for the first 9 alphabets (A-I).
The associated dataset has also been included. To run the script as is, place it in a directory with the data (individual alphabet folders) in "./asl_data/asl_data/".
The model had ~84% test accuracy.'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
use_cuda = torch.cuda.is_available()

#Define model.
class ASL_model(nn.Module):
  def __init__(self):
    super(ASL_model, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)
    self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
    self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
    self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
    self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)
    self.conv7 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
    self.conv8 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
    self.fc1 = nn.Linear(in_features = 25088, out_features = 10000, bias = True)
    self.fc2 = nn.Linear(in_features = 10000, out_features = 500, bias = True)
    self.fc3 = nn.Linear(in_features = 500, out_features = 9, bias = True)
    
    #Xavier Init to deal with dead/saturated units. This is messy but couldn't find a cleaner way.
    nn.init.xavier_uniform(self.conv1.weight)
    nn.init.xavier_uniform(self.conv2.weight)
    nn.init.xavier_uniform(self.conv3.weight)
    nn.init.xavier_uniform(self.conv4.weight)
    nn.init.xavier_uniform(self.conv5.weight)
    nn.init.xavier_uniform(self.conv6.weight)
    nn.init.xavier_uniform(self.conv7.weight)
    nn.init.xavier_uniform(self.conv8.weight)
    nn.init.xavier_uniform(self.fc1.weight)
    nn.init.xavier_uniform(self.fc2.weight)
    nn.init.xavier_uniform(self.fc3.weight)

    self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2) 

  def forward(self, input):
    x = self.max_pool(F.tanh(self.conv1(input)))
    x = self.max_pool(F.relu(self.conv3(F.tanh(self.conv2(x)))))
    x = self.max_pool(F.tanh(self.conv5(F.tanh(self.conv4(x)))))
    x = self.max_pool(F.relu(self.conv7(self.conv6(x))))
    x = self.max_pool(F.tanh(self.conv8(x)))
    x = x.reshape(-1, 25088)
    x = F.tanh(self.fc1(x))
    x = self.fc2(x)
    return self.fc3(x)

if (use_cuda):
#Try using a GPU.
  device = torch.device("cuda:0")
  torch.backends.cudnn.benchmark = True
  print("Using GPU")
else:
  device = "cpu"
torch.backends.cudnn.benchmark = True

#Setting up a custom dataset class. To run the code as is, place all the individual alphabet folders in ./asl_data/asl_data (referenced from this file).

class asl_data(Dataset):
  def __init__(self, dir):
    self.labels = []
    self.inputs = []
    for directory in os.listdir(dir):
      for file in os.listdir(os.path.join(dir, directory)):
        self.labels.append(str(directory))
        self.inputs.append(os.path.join(dir, directory, file))
  def __len__(self):
    return len(self.labels)
  def __getitem__(self, idx):
    return {"input": torch.tensor(np.array(Image.open(self.inputs[idx])), dtype = torch.float32), "label": self.labels[idx]}

#Shuffle and set up the train test and val loaders.
full_data = asl_data("asl_data/asl_data")
indices = list(range(len(full_data)))
np.random.shuffle(indices)

train_sampler = SubsetRandomSampler(indices[600:815])
val_sampler = SubsetRandomSampler(indices[0:200])
test_sampler = SubsetRandomSampler(indices[815:])

train_loader = DataLoader(full_data, batch_size = 25, num_workers = 4, sampler = train_sampler)
val_loader = DataLoader(full_data, batch_size = 15, num_workers = 4, sampler = val_sampler)
test_loader = DataLoader(full_data, batch_size = 1, num_workers = 1, sampler = test_sampler)

#Initialize model and send to the device to use (GPU/CPU).
network = ASL_model()
network.to(torch.device(device))

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I"] #Setup a list of the classes. We will input the index of each class in this list while computing CE loss.

def train_model(model, train_loader, epochs = 35, lr = 0.001, momentum = 0.5):
  #Defining the training function for this model. We use SGD and the default parameters are stated as input args.
  
  optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
  criterion = nn.CrossEntropyLoss()
  train_acc = []
  val_losses = []
  losses = []
  epoch_list = []
  val_epochs = []
  for epoch in range(epochs):
    print("EPOCH: " + str(epoch))
    average_loss = []
    for i, data in enumerate(train_loader):
      optimizer.zero_grad()
      #pred = model(data["input"].cuda())
      pred = model(torch.transpose(torch.transpose(data["input"], 1, 3), 2,3).cuda())
      labels = torch.tensor([classes.index(x) for x in data["label"]]).cuda()
      loss = criterion(pred, labels)
      average_loss.append(loss)
      loss.backward()
      optimizer.step()
    
    print(sum(average_loss)/len(average_loss))
    losses.append(sum(average_loss)/len(average_loss))
    val_losses.append(get_val_loss(model, val_loader))
    val_epochs.append(epoch)
      
    epoch_list.append(epoch)
  #Plotting training and val losses.
  plt.plot(epoch_list, losses)
  #Save the model in ./asl_data as classifier.pt
  torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "optimizer_state_dict": optimizer.state_dict(), "loss": loss}, "./asl_data/classifier.pt" )
  
  def get_val_loss(model, val_loader):
  #Use this function to get the avg val loss for 1 epoch. Used while training.
  criterion = nn.CrossEntropyLoss()
  avg_loss = []
  for data in val_loader:
    pred = model(torch.transpose(torch.transpose(data["input"], 1, 3), 2,3).cuda())
    labels = torch.tensor([classes.index(x) for x in data["label"]]).cuda()
    loss = criterion(pred, labels)
    avg_loss.append(loss)
  return sum(avg_loss)/len(avg_loss)
  plt.plot(val_epochs, val_losses)
  
#Train the model now:
train_model(model, train_loader)

#Function to test the model (returns accuracy on the test set).
def get_test_acc(model, test_loader):
  total = 0
  correct = 0
  for i, data in enumerate(test_loader):
    total += 1
    pred = model(torch.transpose(torch.transpose(data["input"], 1, 3), 2,3).cuda())
    if (int(torch.argmax(pred)) == classes.index(data["label"][0])):
      correct += 1
  return correct/total

#Compute the testing accuracy.
get_test_acc(model, test_loader)
  

