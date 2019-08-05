# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:12:47 2019

@author: Austin Bell
"""
import torch
# use this class to edit all options used within the code
class parameters(object):
    def __init__(self, basePath):
        self.pretrain = False
        self.data = "SICK"
        
        # data file paths
        self.basePath = basePath
        self.inFile = basePath + "/Development/SentenceSimilarity/Data/Inter/Train{}".format(self.data)
        self.testFile = basePath + "/Development/SentenceSimilarity/Data/Inter/Test{}".format(self.data)
        
        
        # pretrained models
        self.bertPath = "C:/Users/Austin Bell/Documents/NLP/pretrainedModels/BERT/uncased_L-12_H-768_A-12"
        self.ftPath = "C:/Users/Austin Bell/Documents/NLP/pretrainedModels/fastText/wiki.en/wiki.en.bin"
        
        # parameters specific to SICK final model
        if self.pretrain == False:
            self.augFile = basePath + "/Development/SentenceSimilarity/Data/Inter/AugmentedSICK"
            self.trainAugFile = basePath + "/Development/SentenceSimilarity/Data/Inter/TrainSickAug"
            self.pretrained_model = basePath + "/Development/SentenceSimilarity/Data/Models/MaLSTM STS.model"
            self.pretrained_vocab = basePath + "/Development/SentenceSimilarity/Data/Models/STS vocab"
            
        
        # Model parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bs = 10
        self.lr = .001
        self.epochs = 100
        self.hidden_dim = 50 
        self.annealing_wgt = .75
        self.reduce_lr_epochs = 4
        self.early_stop = 8
        
        self.model_path = basePath + "/Development/SentenceSimilarity/Data/Models/MaLSTM {}.model".format(self.data)
        self.vocab_path = basePath + "/Development/SentenceSimilarity/Data/Models/{} vocab.txt".format(self.data)
        self.embed_path = basePath + "/Development/SentenceSimilarity/Data/Models/{} embed.npy".format(self.data)
        