# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:47:43 2019

Utility functions for sentence similarity

Data prep section:
    - Tokenize and convert to indices
    - pad sentences 
    - create dataloaders with batched tensors 
        - (sentence1 batch, sentence2 batch, score batch)

@author: Austin Bell
"""
#basePath = "C:/Users/Austin Bell/Documents/NLP/Sentence_Similarity"
#inFile = basePath + "/Data/Inter/TrainSICK"

#sentences = open(inFile, "r", encoding = "utf-8").readlines()

"""
Convert to Bert indices
using huggingface's BERT package: https://github.com/huggingface/pytorch-pretrained-BERT
"""

import torch
from torch.utils import data
import re
import spacy
nlp = spacy.load("en_core_web_sm")


# Converts all tokens to IDS needed for BERT model
def map2idx(sentences, params, vocab = None):
    
    # split --> tokenize --> convert to ids
    # slightly repetitive code, but easier to understand what is going on    
    sMap = list(map(lambda x: (x.split("\t")[0], x.split("\t")[1], x.split("\t")[2]), sentences))
    
  
    # goes tuple by tuple (sentence1, sentence2, score) and converts to spacy
    # then goes word by word and tokenizes using Spacy tokenizer
    sMap = list(map(lambda sentence: (list(map(lambda word: vocab.word2idx(word.norm_), nlp(sentence[0]))),
                                      list(map(lambda word: vocab.word2idx(word.norm_), nlp(sentence[1]))),
                                      re.sub("\n", "", sentence[2])), sMap))
        
    return sMap


"""
Pad each sentence to max length in their group (i.e., first column sentences to max length of sentences from first column)
padding idx = 0
We will reduce padding length to longest sentence in batch when training

This can very easily be expanded to incorporate bi-directionality or characters level
"""

#sMap = map2idx(sentences)

def inputTensors(sMap):
    # get max length across all sentences 
    # they need to be the same as we will compute manhattan distance
    cola_len = max(list(map(lambda x: len(x[0]), sMap)))
    colb_len = max(list(map(lambda x: len(x[1]), sMap)))
    
    # init lists
    padded_sent_a = []
    padded_sent_b = []
    senta_lengths = [] # store original lengths 
    sentb_lengths = []
    gs = [] # gold standard scores
    
    # pad according to max length
    for sentence_a, sentence_b, score in sMap:
        
        # pad sentences
        padded_sent_a.append([0]*(cola_len - len(sentence_a)) + sentence_a)
        padded_sent_b.append( [0]*(colb_len - len(sentence_b)) + sentence_b)
        
        # store sentence lengths and scores
        senta_lengths.append(len(sentence_a))
        sentb_lengths.append(len(sentence_b))
        gs.append(float(score))
        
        
    assert len(padded_sent_a) == len(padded_sent_b) == len(senta_lengths) == len(sentb_lengths) == len(gs)
        
    # Convert to Tensors
    padded_sent_a = torch.LongTensor(padded_sent_a)
    padded_sent_b = torch.LongTensor(padded_sent_b)
    senta_lengths = torch.LongTensor(sentb_lengths)
    sentb_lengths = torch.LongTensor(sentb_lengths)
    gs = torch.FloatTensor(gs)
    
    return padded_sent_a, padded_sent_b, senta_lengths, sentb_lengths, gs
    

#train_inputs = inputTensors(sMap)

# prep dataset to for dataloader
class Dataset(data.Dataset):

    def __init__(self, padded_sent_a, padded_sent_b, senta_lengths, sentb_lengths, gs):
        self.sentence_a = padded_sent_a
        self.sentence_b = padded_sent_b
        self.senta_lengths = senta_lengths
        self.sentb_lengths = sentb_lengths
        self.gs = gs
        
    def __getitem__(self, i):
        return self.sentence_a[i], self.sentence_b[i],  self.senta_lengths[i], self.sentb_lengths[i], self.gs[i]
    
    def __len__(self):
        return len(self.sentence_a)



#train_loader = data.DataLoader(Dataset(*train_inputs), batch_size=10, shuffle = True)
