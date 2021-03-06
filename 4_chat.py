# -*- coding: utf-8 -*-
"""chat.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Al4j5XISuDu1IAjBhCtlxom0EDFT2LMN

###########  4  ###########
"""

# !nvidia-smi 
#https://github.com/python-engineer/pytorch-chatbot

import random
import string # to process standard python strings

import warnings # Hide the warnings
warnings.filterwarnings('ignore')

import torch

import nltk
nltk.download('punkt')

#from google.colab import drive
#drive.mount("/content/drive")

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/My Drive/Colab Notebooks/NLP/ChatBot/"
# !ls

import random
import json

import torch

from 1_model import NeuralNet
from 2_nltk_utils import bag_of_words, tokenize

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
    else:
        print(f"{bot_name}: I do not understand...")





# !pip install wikipedia
# import wikipedia as wk #pip install wikipedia
# from collections import defaultdict

# #step 8: tf-idf calculation
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# ##?? sent_tokens, sentence, input

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
#     reg_ex = re.search('from wikipedia (.*)', input)#tell me about 
#     try:
#         if reg_ex:
#             topic = reg_ex.group(1)
#             wiki = wk.summary(topic, sentences = 3)
#             return wiki
#     except Exception as e:
#             print("No content has been found")

# import re, string, unicodedata

# bot_name = "Sam"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "quit" or sentence == "bye":
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

#     elif sentence == sentence.append("From Wikipedia"):
        

        
#         # sent_tokens.append(sentence)
#         # word_tokens=word_tokens+nltk.word_tokenize(sentence)
#         # final_words=list(set(word_tokens))
#         # print("Sid: ",end="")
#         print(response(sentence))
#         sent_tokens.remove(sentence)

              
          


#     else:
#       #print(response(sentence))
#         print(f"{bot_name}: I do not understand...")
