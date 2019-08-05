# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:11:45 2019

Pre-process SemEval and SICK data into standard datasets
tab separated (Sentence 1, Setence 2, Relatedness score)

@author: Austin Bell
"""
import os, re
import numpy as np

# process STS SemEval 2013 data
def processSTS(path):
    
    inputFiles = ['STS.input.headlines.txt', 'STS.input.OnWN.txt', 'STS.input.FNWN.txt']
    gsFiles = ['STS.gs.headlines.txt', 'STS.gs.OnWN.txt', 'STS.gs.FNWN.txt']
    df = []
    
    for sentences, gs in zip(inputFiles, gsFiles):   
        # get sentences and gs scores
        inSentences = open("{}/Raw/STS/{}".format(path, sentences), "r", encoding = "utf-8").readlines()
        gsScore = open("{}/Raw/STS/{}".format(path, gs), "r", encoding = "utf-8").readlines()
    
        # concat
        concat = [re.sub("\n", "",sentence) + "\t" + gs for sentence, gs in zip(inSentences, gsScore)]
        df += concat
        
    n = int(len(df)*.8)
    trainSTS = df[:n]
    testSTS = df[n:]
    
    # output
    open("{}/Inter/TrainSTS".format(path), "w", encoding = "utf-8").writelines(trainSTS)
    open("{}/Inter/TestSTS".format(path), "w", encoding = "utf-8").writelines(testSTS)
    
# Process SICK Data
def processSICK(path):
    rawSick = open("{}/Raw/SICK.txt".format(path), "r", encoding = "utf-8").readlines()
    rawSick = [list(np.array(col.split("\t"))[[1,2,4,-1]]) for col in rawSick] # split and select relevant columns
    
    # create train and test
    trainSick = [item for item in rawSick if item[3] in ["TRAIN\n", "TRIAL\n"]]
    testSick = [item for item in rawSick if item[3] in ["TEST\n"]]
    
    # simplify and output
    trainSick = ["{}\t{}\t{}\n".format(row[0], row[1], row[2]) for row in trainSick]
    open("{}/Inter/TrainSICK".format(path), "w", encoding = "utf-8").writelines(trainSick)
    
    testSick = ["{}\t{}\t{}\n".format(row[0], row[1], row[2]) for row in testSick]
    open("{}/Inter/TestSICK".format(path), "w", encoding = "utf-8").writelines(testSick)
 