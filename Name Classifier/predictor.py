#The driver code for the RNN mdoel.
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
all_letters = "0" + string.ascii_letters + " .,;'"


def encode_name(name):
    #Returns sequence of one-hot-encoded vectors based on characters in a string (name)
    encoding = torch.zeros(len(name), len(all_letters))
    for i, alphabet in enumerate(name):
        encoding[i][all_letters.find(alphabet)] = 1
    return encoding

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

#Given a name, use the model to determine the odds that it has a certain origin
def predict(name, model):
    hidden = (torch.zeros(3, 1, 80).to(torch.device(device)), torch.zeros(3, 1, 80).to(torch.device(device)))
    lang = ["Russian", "Indian", "English", "Oriental"]
    name = encode_name((19 - len(name)) * "0" + name).unsqueeze(0)
    forward = model(name.to(torch.device(device)), hidden) 
    answer = list(F.softmax(forward) * 100)[0]

    for i in range(len(answer)):
        print(lang[i] + " : " + str(round(float(answer[i]), 2)) + "%")

model = name_predictor(len(all_letters), 80, 4)
checkpoint = torch.load("./name_classifier.pt")
model.load_state_dict(checkpoint["model_state_dict"])
predict("Xiao", model)