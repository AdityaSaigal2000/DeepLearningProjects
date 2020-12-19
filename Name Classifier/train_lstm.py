#Classifies a last name into 4 categories: Russian, Indian, English, Oriental
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import json
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import unicodedata
import numpy as np
import matplotlib.pyplot as plt 
from Levenshtein import distance

all_letters = "0" + string.ascii_letters + " .,;'" #Uppercase and lowercase letters + some punctuation + 0's for padding
def unicodeToAscii(s):
    #Converts non-English characters in unicode to their equivalent in ASCII
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)

def encode_name(name):
    #Returns sequence of one-hot-encoded vectors based on characters in a string (name)
    encoding = torch.zeros(len(name), len(all_letters))
    for i, alphabet in enumerate(name):
        encoding[i][all_letters.find(alphabet)] = 1
    return encoding

def collate_fn(batch):
    #Places multiple inputs and targets into a tuple while batching rather than stacking them (helps when inputs have different dimesions)
    return tuple(zip(*batch))

class name_data(Dataset):
    #Custom dataset class for names.
    def __init__(self, dict):
        self.max = 0
        self.data = []
        for lang in dict:
          for name in dict[lang]:
            if(len(unicodeToAscii(name)) > self.max):
              self.max = len(unicodeToAscii(name))
              self.max_name = unicodeToAscii(name)
            self.data.append((unicodeToAscii(name), lang))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        padded_name = (self.max - len(self.data[idx][0]))*"0" + self.data[idx][0]
        return (encode_name(padded_name), self.data[idx][1]) 

class name_predictor(nn.Module):
    #Defining the RNN architecture.
    def __init__(self, input_dim, hidden_dim, num_targets):
        super(name_predictor, self).__init__()
        # Using 3 LSTM layers. Feed the output at the last time step to 2 fully connected layers.
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = 3, batch_first = True)
        self.dense1 = nn.Linear(in_features = hidden_dim, out_features = 20, bias = True)
        self.dense2 = nn.Linear(in_features = 20, out_features = num_targets, bias = True)

        #Using Xavier initialization on the FC layers and initializing the LSTM parmaters from a normal dist. Tried other initializations but this worked the best.
        for param in self.lstm.parameters():
            torch.nn.init.normal(param.data)

        torch.nn.init.xavier_uniform(self.dense1.weight)
        torch.nn.init.xavier_uniform(self.dense2.weight)

    def forward(self, names, hidden):
        return self.dense2(F.relu(self.dense1(F.relu(self.lstm(names, hidden)[1][1][0]))))

with open("name_data.json", "r") as file:
  data = json.load(file)
  new_data = {}
#Specifying that an Oriental last name can belong to any of the following classes
data["Oriental"] = data["Chinese"] + data["Japanese"] + data["Korean"] + data["Vietnamese"]
for key in data:
    if(key == "Russian" or key == "Indian" or key == "English" or key == "Oriental"):
        new_data[key] = data[key]

#The input data only has 4 classes now.
data = new_data

#Sampling Russian and English last names (there is a significant overrepresentation of these classes in the dataset). Uisng Levenshtein distance to sample the most diverse set of names.
under_sampled_dict = {}
for lang in data:
    under_sampled_dict[lang] = []
    if(len(data[lang]) > 1500):
        for i in range(1500):
            min_dist = 0
            if(under_sampled_dict[lang]):
                for name in data[lang]:
                    candidate_dist = min([distance(name, j) for j in under_sampled_dict[lang]])
                    if(candidate_dist > min_dist and not name in under_sampled_dict[lang]):
                        replace = name
                        min_dist = candidate_dist
                under_sampled_dict[lang].append(replace)
            else:
                under_sampled_dict[lang].append(data[lang][0])
            if(len(under_sampled_dict[lang]) > 1):
                if(under_sampled_dict[lang][-1] == under_sampled_dict[lang][-2]):
                    under_sampled_dict[lang] = under_sampled_dict[lang][:-1]
            #print(under_sampled_dict)
    else:
        under_sampled_dict[lang]  = data[lang]

data = new_dict
#Setting up the train, val and test loaders
full_data = name_data(data)
indices = list(range(len(full_data)))
np.random.shuffle(indices)
train_sampler = SubsetRandomSampler(indices[0:5000])
val_sampler = SubsetRandomSampler(indices[5000:5500])
test_sampler = SubsetRandomSampler(indices[5500:])
train_loader = DataLoader(full_data, batch_size = 500, num_workers = 4, sampler = train_sampler, collate_fn = collate_fn)
val_loader = DataLoader(full_data, batch_size = 500, num_workers = 4, sampler = val_sampler)
test_loader = DataLoader(full_data, batch_size = 1, num_workers = 1, sampler = test_sampler)

#Initializing the model with an input dim of len(all_letters), an LSTM dim of 80 and an output dim of 4 (num_targets)
model = name_predictor(len(all_letters), 80, 4)
use_cuda = True #Use GPU
if (use_cuda):
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    print("Using GPU")
else:
    device = "cpu"

CUDA_LAUNCH_BLOCKING = 1 #Need this for proper error messages when running on the GPU

model.to(torch.device(device)) #Send model to GPU

target_encoding = {"Russian" : 0, "Indian" : 1, "English" : 2, "Oriental" : 3}

#Training function
def train(model, train_loader, val_loader, epochs = 100, lr = 0.02, weight_decay = 0.05, momentum = 0.9, batch_size = 500):
    #Train for 100 epochs
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss()
    losses = []
    val_losses = []
    epoch_list = []

    for epoch in range(epochs):
        if(epoch == 50):
            #Decay learning rate
            optimizer = torch.optim.SGD(model.parameters(),  lr = 0.25*lr, momentum = momentum, weight_decay = weight_decay)
        epoch_list.append(epoch)
        avg_loss = [] #Avg loss for this epoch
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            targets = torch.tensor([target_encoding[language] for language in data[1]], dtype = torch.long).to(torch.device(device))
            hidden = (torch.zeros(3, batch_size, 80).to(torch.device(device)), torch.zeros(3, batch_size, 80).to(torch.device(device)))
            forward = model(torch.stack(data[0]).to(torch.device(device)), hidden)
            loss = criterion(forward, targets)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss)
        print(epoch)  
        print(sum(avg_loss)/len(avg_loss))
        val_loss = compute_val_loss(model, val_loader)
        print(val_loss)
        losses.append(sum(avg_loss)/len(avg_loss))
        val_losses.append(val_loss)
        plt.plot(epoch_list, losses)
        plt.plot(epoch_list, val_losses)

def compute_val_loss(model, val_loader):
    #Returns the average loss on the validation set for every epoch
    avg_loss = []
    criterion = nn.CrossEntropyLoss()
    for data in val_loader:
        with torch.no_grad():
            hidden = (torch.zeros(3, 500, 80).to(torch.device(device)), torch.zeros(3, 500, 80).to(torch.device(device)))
            #print(data[0].shape)
            forward = model(data[0].to(torch.device(device)), hidden)
            target = torch.tensor([arabic_indian_encoding[language] for language in data[1]], dtype = torch.long).to(torch.device(device))
            #print(target)
            #print(forward)
            avg_loss.append(criterion(forward[0], target))
    return sum(avg_loss)/len(avg_loss)

#Train the model
train(model, train_loader, val_loader)
torch.save({"model_state_dict": model.state_dict()}, "./name_classifier.pt" ) #Save the model
#Now test the model accuracy:
correct = 0
total = 0
results = {"English": 0, "Indian" : 0, "Russian" : 0, "Oriental" : 0}

with torch.no_grad():
    for data in test_loader:
        hidden = (torch.zeros(3, 1, 80).to(torch.device(device)), torch.zeros(3, 1, 80).to(torch.device(device)))
        forward = model(data[0].to(torch.device(device)), hidden)
        answer = torch.argmax(forward)
        if(answer == target_encoding[data[1][0]]):
          correct += 1
        else:
          results[data[1][0]] += 1
        total += 1

print("Test Accuracy : " + str(correct/total))
print("This dictionary shows the number of errors per class: ")
print(results) 