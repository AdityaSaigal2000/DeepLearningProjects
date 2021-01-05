import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import string
import unicodedata
import random
from nltk.translate.bleu_score import sentence_bleu
from architecture import *

use_cuda = torch.cuda.is_available()
if (use_cuda):
    device = torch.device('cuda:0') 
else:
    device = torch.device("cpu")

print("CUDA: ", use_cuda)
print(device)

torch.manual_seed(0) #For reproducibility

#Data processing:
def processLanguages(lang1, lang2):
    #Given 2 languages (and a file with the data in the same directory, this function returns a shuffled dict of lists of the sentences in both languages)
    with open(lang1 + "-" + lang2 + ".txt", "r") as file:
        lines = file.readlines()

    langs = {lang1 : [], lang2 : []}
    
    for line in lines:
        lang = " ".join(unicodeToAscii(line.split("\t")[0]).split()) 
        
        if (not lang in langs[lang1]):
            langs[lang1].append(lang)
            langs[lang2].append(" ".join(unicodeToAscii(line.split("\t")[1][:-1]).split()))
    
    for i in range(len(langs[lang1])):
        last = langs[lang1][i][-1]
    
        if (last == "." or last == "!" or last == "?"):
            langs[lang1][i] = langs[lang1][i][:-1]

        last = langs[lang2][i][-1]
        if (last == "." or last == "!" or last == "?"):
            langs[lang2][i] = langs[lang2][i][:-1]
    #langs[lang1] = langs[lang1][0:1500] + langs[lang1][6000:7500] + langs[lang1][15000:16500] + langs[lang1][-1500:-1]
    #langs[lang2] = langs[lang2][0:1500] + langs[lang2][6000:7500] + langs[lang2][15000:16500] + langs[lang2][-1500:-1]
    shuffle = list(zip(langs[lang1], langs[lang2]))
    random.shuffle(shuffle)
    langs[lang1], langs[lang2] = zip(*shuffle)
    return langs

def unicodeToAscii(s):
    #Get rid of fancy characters and accents.
    all_letters = string.ascii_letters + string.digits + " .,;'"
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)

def word2idx(sents):
    #For a specific language, assing an index to each word and store in a dictionary.
    idx = 0
    word_dict = {}
    for sent in sents:
        for word in sent.split():
            if(not word in word_dict):
                word_dict[word] = idx
                idx +=1 
    return word_dict

def train(model, train_loader, batch_size, num_epochs, lr, momentum, weight_decay, tf_ratio, output_embedding_size):
    #Function to train the entire translator.
    model.to(torch.device(device))
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay) 
    epochs = []
    losses = []
    tf_ratio = tf_ratio
    for epoch in range(num_epochs):
        print(epoch)
        avg_loss = []
        epochs.append(epoch)
        #We randomly choose from a uniform distribution if Teacher Forcing should be used for a certain epoch
        use_tf = np.random.uniform(0, 1) < tf_ratio
        print(use_tf)

        for mini_batch in train_loader:
            optimizer.zero_grad()
            loss_val = 0
            #The initial embedding for any minibatch is that of the <SOS> token.
            prev_embedding = model.output_embeddings(torch.tensor(model.output_word2idx["<SOS>"], dtype = torch.long).to(torch.device(device))).expand(batch_size, 1, output_embedding_size)
            prev_hidden = None #We pass in the prev_hidden value as None to the model. This signals to the model that it must encode this minibatch and store it in a class variable. It also 
            #tells it to initialize the hidden state from the hidden initialization FC layer.
            input_embed, input_lengths, output_embed, output_lengths, output_sents = model.embed_batch(mini_batch) #Start passing the batch through the model
            word_list = [output_sents[j].split() for j in range(len(output_sents))] #Process the targets 
            #Predict one word at a time
            for i in range(len(output_sents[0].split())):
                output, hidden = model(input_lengths, input_embed, prev_hidden, prev_embedding)
                prev_hidden = hidden
                output_idxs = torch.argmax(output, dim = 2)
                if (use_tf):
                  prev_embedding = output_embed[:, i, :].unsqueeze(1).to(torch.device(device))
                else:
                  prev_embedding = model.output_embeddings(output_idxs)
                targets = torch.tensor([model.output_word2idx[word_list[j][i]] for j in range(len(word_list))])
                #Need to do this to ensure that none of the padding tokens in a sentence are being used while computing the loss function (mask all the output indices that are not the target index to -inf and set the correct one to 1)
                for i in range(len(targets)):
                    if (int(targets[i]) == len(model.output_word2idx) - 1):
                        targets[i] = torch.argmax(output[i][0]) 
                        output[i][0] = torch.tensor([-1 *float("Inf")] * output[i][0].shape[0])
                        output[i][0][targets[i]] = 1
            output = output.squeeze(1)
            loss_val += loss(output, targets.to(torch.device(device)))

            loss_val.backward()
            optimizer.step()
            avg_loss.append(loss_val)
        losses.append(sum(avg_loss)/len(avg_loss))
        print(sum(avg_loss)/len(avg_loss))

    plt.plot(epochs, losses)

def predict(model, sentence):
    #Translate a French sentence given a trained model
    input_embedding = torch.tensor([list(model.input_embeddings(torch.tensor(model.input_word2idx[x], dtype = torch.long))) for x in sentence.split()]).unsqueeze(0)
    input_lengths = [len(sentence.split())]
    prev_hidden = None
    prev_embed = model.output_embeddings(torch.tensor(model.output_word2idx["<SOS>"], dtype = torch.long).to(torch.device(device))).unsqueeze(0).unsqueeze(0)
    output_ls = []
    current_word = ""
    #Keep passing the previous word through the model until it produces an <EOS> token
    while not current_word == "<EOS>":
        output, hidden = model(input_lengths, input_embedding, prev_hidden, prev_embed)
        current_word = list(model.output_word2idx.keys())[torch.argmax(output, dim = 2)]
        prev_hidden = hidden
        prev_embed = model.output_embeddings(torch.argmax(output, dim = 2))
        if (not current_word == "<EOS>"):
            output_ls.append(current_word)
    return " ".join(output_ls)

#Setting up data for French to English translation.
langs = processLanguages("eng", "fra")
eng_words_idx = word2idx(langs["eng"])
fra_words_idx = word2idx(langs["fra"])

#Adding EOS and SOS tokens to the language dictionaries.
eng_words_idx["<EOS>"] = len(eng_words_idx)
eng_words_idx["<SOS>"] = len(eng_words_idx)
eng_words_idx["<PAD>"] = len(eng_words_idx)
fra_words_idx["<EOS>"] = len(fra_words_idx)
fra_words_idx["<SOS>"] = len(fra_words_idx)
fra_words_idx["<PAD>"] = len(fra_words_idx)

#Define a model based on the number of words in the corpus (we randomly sampled 6000 sentences from the dataset)
model = Translator(7349, 256, 512, 512, 5561, 256, fra_words_idx, eng_words_idx)
data = Sentences(langs)
train_sampler = SubsetRandomSampler(indices[0:6000])

#Setup the train loader
train_loader = DataLoader(data, batch_size = 400, num_workers = 4, sampler = train_sampler)

#Train the model
train(model, train_loader, 400, 10, 0.002, 0.9, 0.03, 0.5, 256)
torch.save({"model_state_dict": model.state_dict()}, "./translator.pt" ) #Save the model


#Evaluate the trained model by making predictions on the training set and calculating the BLEU scores for each sentence.
bleu = 0
for i in range(6000):
    ans = predict(model, langs["fra"][i]).split()
    bleu += sentence_bleu([langs["eng"][i].split()], ans)

#Print the BLEU Score on the traning set
bleu = bleu/6000

