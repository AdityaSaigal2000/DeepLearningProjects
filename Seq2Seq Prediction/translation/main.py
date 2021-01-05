import torch
import architecture
from architecture import Translator 
from train import predict, word2idx
import json

with open("lang_data.json", "r") as file:
    langs = json.load(file)

eng_words_idx = word2idx(langs["eng"])
fra_words_idx = word2idx(langs["fra"])

#Adding EOS and SOS tokens to the language dictionaries.
eng_words_idx["<EOS>"] = len(eng_words_idx)
eng_words_idx["<SOS>"] = len(eng_words_idx)
eng_words_idx["<PAD>"] = len(eng_words_idx)
fra_words_idx["<EOS>"] = len(fra_words_idx)
fra_words_idx["<SOS>"] = len(fra_words_idx)
fra_words_idx["<PAD>"] = len(fra_words_idx)

model = Translator(7349, 256, 512, 512, 5561, 256, fra_words_idx, eng_words_idx)
checkpoint = torch.load("./translator.pt", map_location = torch.device(architecture.device))
model.load_state_dict(checkpoint["model_state_dict"])

print("Write a sentence in French and the model will try to translate it to English: ")
while True:
    french_sent = input()
    if(not french_sent == "Quit"):
        print("\n" + predict(model, french_sent))
    else:
        exit()
    print("\nWrite another sentence or write 'Quit' to exit.")
