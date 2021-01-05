#Includes classes wrapping the model architecture and the dataset that will be used with the dataloader.
#Model architecture is the one proposed in "Neural machine translation by jointly learning to align and translate". Uses attention to improve performance on longer sentences.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import itertools
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore", category = DeprecationWarning)

use_cuda = torch.cuda.is_available()
if (use_cuda):
    device = torch.device('cuda:0') 
else:
    device = torch.device("cpu")

class Sentences(Dataset):
    #Will be used to setup the dataloader for the sentences.
    def __init__(self, langs):
        self.langs = langs
        self.input_lang, self.output_lang = list(langs.keys())[1], list(langs.keys())[0]
    def __getitem__(self, idx):
        return (self.langs[self.input_lang][idx], len(self.langs[self.input_lang][idx].split())), (self.langs[self.output_lang][idx], len(self.langs[self.output_lang][idx].split()))
    def __len__(self):
        return len(self.langs[self.input_lang])

#Defining the model architecture: 1st make Encoder and Decoder seperately. Then merge them together in a single class.
class Encoder(nn.Module):   
  def __init__(self, input_dim, hidden_dim, num_layers):     
    super(Encoder, self).__init__()     
    #Uses a bidirectional GRU and outputs the hidden states in both directions.
    self.GRU = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first = True, bidirectional = True)
  def forward(self, x, lengths):
    gru_input = pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted = False).to(torch.device(device))
    output, h_n = self.GRU(gru_input)
    output = pad_packed_sequence(output, batch_first = True)
    h_n = torch.cat((h_n[0], h_n[1]), dim = 1)
    return output[0], h_n.unsqueeze(0)

class Decoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_size, output_embedding_dim):
    super(Decoder, self).__init__()
    #Uses attention mechanism: computes the attention weights by passing the previous hidden state of the decoder and the hidden states of the encoder through a linear layer
    #Then performs an inner product between the hidden encoder states and the attention weights. This is our context vector that we feed in with the previous state into the GRU.
    self.GRU = nn.GRU(input_size = input_dim + output_embedding_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first = True)
    self.attn = nn.Linear(input_dim + hidden_dim, 1)
    self.dense = nn.Linear(hidden_dim, output_size)
  
  def forward(self, encoder_hidden, prev_hidden, prev_embed, max_len):
    hidden = prev_hidden.squeeze(0).unsqueeze(1)
    hidden = hidden.expand(hidden.shape[0], max_len, hidden.shape[-1])
    attn_input = torch.cat((encoder_hidden, hidden), 2)
    attn_weights = self.attn(attn_input)
    attn_probs = F.softmax(attn_weights, 1)
    attn_probs = attn_probs.reshape(attn_probs.shape[0], attn_probs.shape[2], attn_probs.shape[1])
    context_vec = torch.bmm(attn_probs, encoder_hidden)
    rnn_input = torch.cat((context_vec, prev_embed), 2)
    output, h_n = self.GRU(rnn_input, prev_hidden)
    output = self.dense(output)
    return output, h_n

class Translator(nn.Module):
  def __init__(self, input_vocab_size, input_dim, encoder_dim, decoder_dim, output_size, output_dim, input_word2idx, output_word2idx, encoder_layers = 1, decoder_layers = 1):
    #Wraps the input embeddings, encoder, decoder and output embeddings into a single class. With this, we can train the entire model end to end.
    super(Translator, self).__init__()
    self.input_word2idx = input_word2idx
    self.output_word2idx = output_word2idx
    self.input_embeddings = nn.Embedding(input_vocab_size, input_dim)
    self.output_embeddings = nn.Embedding(output_size, output_dim)
    self.decoder_initialize = nn.Linear(2 * encoder_dim, decoder_dim)
    self.encoder = Encoder(input_dim, encoder_dim, encoder_layers)
    self.decoder = Decoder(2 * encoder_dim, decoder_dim, decoder_layers, output_size, output_dim)
    self.encoded = None
    self.encoded_hidden = None
    '''for param in self.encoder.parameters():
      try:
        torch.nn.init.xavier_uniform(param.data)
      except:
        torch.nn.init.normal(param.data)

    for param in self.decoder.parameters():
      try:
        torch.nn.init.xavier_uniform(param.data)
      except:
        torch.nn.init.normal(param.data)'''

  def embed(self, batch, word2idx, embedding):
    first_done = False
    lengths = batch[1].tolist()
    max_len = max(batch[1])
    sentences = []
    for i in range(len(batch[0])):
        sent = batch[0][i] + " <EOS>" + " <PAD>" * (max_len - batch[1][i])
        sentences.append(sent)
        if (not first_done):
            emb = torch.tensor([list(embedding(torch.tensor(word2idx[word], dtype = torch.long).to(torch.device(device)))) for word in sent.split()]).unsqueeze(0)
            first_done = True
        else:
            new_sent = torch.tensor([list(embedding(torch.tensor(word2idx[word], dtype = torch.long).to(torch.device(device)))) for word in sent.split()]).unsqueeze(0)
            emb = torch.cat((emb, new_sent), 0)
    return emb, lengths, sentences

  def embed_batch(self, batch):
    batch_input = batch[0]
    batch_output = batch[1]
    input_embed, input_lengths, input_sents = self.embed(batch_input, self.input_word2idx, self.input_embeddings)
    output_embed, output_lengths, output_sents = self.embed(batch_output, self.output_word2idx, self.output_embeddings)
    return input_embed, input_lengths, output_embed, output_lengths, output_sents

  def forward(self, input_lengths, input_embed, prev_hidden, prev_embed):
    #Data flow: In training or prediction functions we compute the embeddings from the sentence by using the class functions. We then pass the embeddings into the forward function
    #to make a prediction.
    if (prev_hidden is None):
      self.encoded_output, self.encoded_hidden = self.encoder(input_embed, input_lengths)
      prev_hidden = self.decoder_initialize(self.encoded_hidden)
    decoded_output, decoded_hidden = self.decoder(self.encoded_output, prev_hidden, prev_embed, max(input_lengths))
    return decoded_output, decoded_hidden

#-------------- USED SPECIFICALLY FOR TESTING --------------------------

#Generate some synthetic data to see if model can be fit to solve a simple problem (string reversal):

'''class syn_data(Dataset):
  def __init__(self):
    self.chars = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    self.inputs = [" ".join(x)  for x in list(itertools.permutations(self.chars))]
    self.outputs = [" ".join(x[::-1]) for x in self.inputs]
  
  def __getitem__(self, idx):
    return (self.inputs[idx], 10), (self.outputs[idx], 9)

  def __len__(self):
    return len(self.inputs)

test_syn = syn_data()
indices = list(range(len(test_syn)))
train_sampler = SubsetRandomSampler(indices[0:100])
data_syn = DataLoader(test_syn, batch_size = 50, num_workers = 4, sampler = train_sampler)
char_to_idx = {x: int(x) - 1 for x in test_syn.chars}
char_to_idx["<EOS>"] = len(char_to_idx)
char_to_idx["<SOS>"] = len(char_to_idx)'''