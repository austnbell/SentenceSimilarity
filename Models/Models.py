# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:26:30 2019

Script that includes the model class
variant of the Siamese LSTM from "Siamese Recurrent Architectures for Learning Sentence Similarity" by Mueller and Thyagarajan
First we pre-train a basic siamese LSTM on STS dataset
- save the hidden states, which are used to initialize the LSTM hidden states in core model
- We first initialize our LSTM weights with small random Gaussian entries 
    (and a separate large value of 2.5 for the forget gate bias to facilitate modeling of long range dependence).


A batch (sentence1, sentence2, and scores) enters model:
    - convert to embedding
    - enter each embedding state in their respective LSTM network 
    - compute manhattan distance between two resulting hidden states
    - minimize mean squared error

@author: Austin Bell
"""


import torch, re
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel,BertPreTrainedModel, BertConfig
from torch.autograd import Variable
import numpy as np


class SiameseFTnetwork(nn.Module):
    def __init__(self, vocab_sz, embed_dim, vectors, params):
        super(SiameseFTnetwork, self).__init__()
        self.params = params
        self.bs = params.bs
        self.vocab_sz = vocab_sz
        self.embed_dim = embed_dim
        self.hidden_dim = params.hidden_dim
        self.device = params.device
        self.pretrain = params.pretrain
        if params.pretrain == False:
            self.trained_state_dict = torch.load(params.pretrained_model)
        
        # siamese embedding layers
        self.embedding = nn.Embedding(vocab_sz, embed_dim, padding_idx = 0)
        self.embedding.weight = nn.Parameter(vectors)
        self.embedding.weight.requires_grad = True
        
        # siamese LSTM
        self.LSTM = nn.LSTM(embed_dim, params.hidden_dim, num_layers = 1, batch_first = True)
        #self.LSTM_b = nn.LSTM(embed_dim, params.hidden_dim, num_layers = 1, batch_first = True)
        
        # linear layer
        #self.siamese2class = nn.Linear()
        
        # initialize parameters
        self.LSTM.load_state_dict(self.init_params())
        #self.LSTM_b.load_state_dict(self.init_params("b"))
        
        
    # init hidden
    # I need to initialize the weights not the hidden states
    def init_hidden(self, bs_at_moment): 
        # initialize with random gaussian entries
        h0_a = Variable(torch.zeros(1, bs_at_moment, self.hidden_dim)).to(self.device)
        c0_a = Variable(torch.zeros(1, bs_at_moment, self.hidden_dim)).to(self.device)
        
        h0_b = Variable(torch.zeros(1, bs_at_moment, self.hidden_dim)).to(self.device)
        c0_b = Variable(torch.zeros(1, bs_at_moment, self.hidden_dim)).to(self.device)
            
        return (h0_a, c0_a), (h0_b, c0_b)   
    

    def init_params(self): 
        #biases are ordered as ingate, forgetgate, cellgate, outgate. 
        state_dict = self.LSTM.state_dict()
        
        if self.params.pretrain == True:

            """
            Update the forget gate bias for each of the LSTMs
            increasing the forget get bias allows for the LSTM to more effectively capture long term dependencies
            so given 200 parameter bias vector, the forget gate will be 50-100
            
            all other params initialized to random gaussian entries
            """
            for weight in state_dict:
                if "bias" in weight:
                    bias = state_dict[weight]
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    
                    fill = np.random.normal(2.5, .25, 50)
                    state_dict[weight][start:end] = torch.tensor(fill)
                    
            return state_dict
        
        elif self.params.pretrain == False:
            """
            initializes the LSTM model with the correct weights from pretrained model
            """
            for name, param in self.trained_state_dict.items():
                if "LSTM_"+lstm_version in name:
                    name = re.sub("LSTM_%s." % lstm_version, "", name)
                    # replace the weight/bias in current state_dict
                    state_dict[name].copy_(param)
                    
            return state_dict
            
    
    def exponent_neg_manhattan_distance(self, hidden1, hidden2):
        return torch.exp(-torch.sum(torch.abs(hidden1-hidden2),dim = 1))
    
    def forward(self, sentence_a, sentence_b, training = True):
        
        # input sentences into respective embedding layers
        embedded_a = self.embedding(sentence_a)
        embedded_b = self.embedding(sentence_b)
          
        # Initialize LSTM embeddings and change forget gate bias
        a0, b0 = self.init_hidden(sentence_a.size(0))
        
        
        # pass output through each LSTM
        _, lstm_a_hidden = self.LSTM(embedded_a, a0)
        _, lstm_b_hidden = self.LSTM(embedded_b, b0)        
        
        # Compute sentence similarity using exp(−||h(a) − h(b)||) L1 between sentence vectors (i.e., LSTM hidden states)
        encoding_a = lstm_a_hidden[0].squeeze()
        encoding_b = lstm_b_hidden[0].squeeze()
        
        if training == False:
            # hidden states will be used for search application
            return encoding_a, encoding_b
        
        L1_loss = self.exponent_neg_manhattan_distance(encoding_a, encoding_b)
        
        # return L1 Distance Calculation 
        return L1_loss*5
        


