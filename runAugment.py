# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:46:56 2019

@author: Austin Bell
"""
basePath = "C:/Users/Austin Bell/Documents/NLP/Sentence_Similarity"
    
import sys
sys.path.append(basePath)

from AugmentData import *

inFile = basePath + "/Data/Inter/TrainSICK"
augFile = basePath + "/Data/Inter/AugmentedSICK"
outFile = basePath + "/Data/Inter/TrainSickAug"
augmentor = dataAugmentor(inFile, outFile, augFile)
augmentor.augment()
        