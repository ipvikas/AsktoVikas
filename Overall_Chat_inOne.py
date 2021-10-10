#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st

###########  1  ###########

#https://www.youtube.com/watch?v=RpWeNzfSUHw&list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg
#intents.json --> nltk_utils.py --> model.py --> train.ipynb --> chat.ipynb --> 
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

# print(tokenize('Hello how are you'))

def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

# print(stem('organize'))

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# print(bag_of_words('Hello how are you', 'hi'))

# import wikipedia as wk #pip install wikipedia
# from collections import defaultdict



# #step9: Cosine Similarity
# # Generating response
# def response(sentence):
#     robo_response=''
#     TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
#     tfidf = TfidfVec.fit_transform(sent_tokens)
#     vals = cosine_similarity(tfidf[-1], tfidf)
#     idx=vals.argsort()[0][-2]
#     flat = vals.flatten()
#     flat.sort()
#     req_tfidf = flat[-2]

#     if(req_tfidf==0) or "From Wikipedia" in sentence:#tell me about
#         print("Checking Wikipedia")
#         if sentence:
#             robo_response = wikipedia_data(sentence)
#             return robo_response
        
        
        
#     else:
#         robo_response = robo_response+sent_tokens[idx]
#         return robo_response



# import re, string, unicodedata
# def wikipedia_data(sentence):
#     reg_ex = re.search('from wikipedia (.*)', sentence)#tell me about 
#     try:
#         if reg_ex:
#             topic = reg_ex.group(1)
#             wiki = wk.summary(topic, sentences = 3)
#             return wiki
#     except Exception as e:
#             print("No content has been found")

###########  2  ###########

import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

###########  3  ###########

import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#2. Loading our JSON Data

from google.colab import drive
drive.mount('/content/drive')

path = 'https://github.com/ipvikas/AsktoVikas/blob/main/intents.json'

import json
with open(path, 'r') as f:
    intents = json.load(f)

# print(intents)



import nltk
nltk.download('punkt')

# from nltk_utils import bag_of_words, tokenize, stem

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))



# stem and lower each word
# ignore_words = ['?', '.', '!']
ignore_words = ['(',')','-',':',',',"'s",'!',':',"'","''",'--','.',':','?',';''[',']','``','o','’','“','”','”','[',';']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))


print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

import torch
import torch.nn as nn

# from model import NeuralNet

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

# !nvidia-smi 
#https://github.com/python-engineer/pytorch-chatbot

import random
import string # to process standard python strings

import warnings # Hide the warnings
warnings.filterwarnings('ignore')

import torch

import nltk
nltk.download('punkt')

# from google.colab import drive
# drive.mount("/content/drive")

import random
import json

import torch

# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
  intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

###################################

# print("Let's chat! (type 'quit' to exit)")
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")




@st.cache(allow_output_mutation=True, suppress_st_warning=True)
st.title('Hindi to English Translator')
text = st.text_area("Enter Text:", value='', height=None, max_chars=None, key=None)
if st.button('Translate to English'):
    if text == '':
        
        st.write('Please enter Hindi text for translation') 
    else:
        
        print("Let's chat! (type 'quit' to exit)")
        while True:
            # sentence = "do you use credit cards?"
            sentence = input("You: ")
            if sentence == "quit":
                break

            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        st.write('', str(out).strip('][\''))
            else:
                print(f"{bot_name}: I do not understand...")

else: pass
