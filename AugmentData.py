# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:20:20 2019

Augment the SICK dataset using Synonyms from MyThes Project
inspired by: Character-level Convolutional Networks for Text Classification (Zhang, Zhao, and LeCunn)

Process
- Load in Sick data and go line by line
- tokenize and get POS for each word using spacy (replace with equivalent wordnet tag)
- get synsets from wordnet (ordered by likelihood of appearance)
- identify replacements using geometric distribution
- select best fit using BERT language model

inputs 
- tab delimited file formatted as (sentence 1, sentence 2, score)

output
- tab delimited file with same format
- appended augmented sentences


Synonyms from word net for data augmentation are imperfect.
WSD is a difficult problem so there is trouble in selecting the correct synset
Wordnet only works with lemmatized words, so replaced synonyms are likely to be in the incorrect tense
    - future work can include using a language model to predict the replacement word rather than synsets
        e.g., randomly mask a noun, adj, verb, or adv token and use BERT lm to predict the word
    
    
Next step: we should switch the exact same word in both sentences to ensure that the equality stays the same
    - this means identifying shared words between sentences
    - only finding synonyms for these
    - make the same replacement to both sentences

@author: Austin Bell
"""
import spacy
nlp = spacy.load("en")
STOP_WORDS = nlp.Defaults.stop_words

from nltk.corpus import wordnet as wn
import re, random
import numpy as np
from collections import OrderedDict
from BertLM import *

class dataAugmentor(object):
    
    def __init__(self, inFile, outFile, augFile, n_comparisons = 5):
        self.inFile = inFile
        self.outFile = outFile
        self.augFile = augFile # augmented data export only
        self.n_comparisons = n_comparisons # number of candidate sentences to choose from 
        self.lm = bertLM() # note that BERT is not deterministic due to its masking method so results may change
        self.sentences = open(inFile, "r", encoding = "utf-8").readlines()    
       
        
    # Core functions that runs data augmentor
    def augment(self):
        global sentence2, augmented2, tuple2
        augmented_data = []
        
        for i, line in enumerate(self.sentences):
            sentence1, sentence2, score = line.split("\t") # split sentence
            sentence1, sentence2 = nlp(sentence1), nlp(sentence2)
        
            # format both sentences (word, lemma, pos)
            tuple1, tuple2 = self.formatLine(sentence1), self.formatLine(sentence2)
            
            # identify set of synonyms
            tuple1, tuple2 = self.wsdSynSet(tuple1, sentence1), self.wsdSynSet(tuple2, sentence2)
            
            
            # replace the sentences with synonyms randomly
            augmented1 = self.replace_synonyms(tuple1, sentence1.text)
            augmented2 = self.replace_synonyms(tuple2, sentence2.text)
            augmented_data += augmented1 + "\t" + augmented2 + "\t" + score
            
            if i % 50 == 0:
                print("{} out of {}".format(i, len(self.sentences)))
            
        # export datafiles
        open(self.augFile, "w", encoding = "utf-8").writelines(augmented_data)
        self.sentences += augmented_data
        open(self.outFile, "w", encoding = "utf-8").writelines(self.sentences)
        
            
                 
    def formatLine(self, sentence):
        # for each token in sentence create tuple (word, lemma, pos)
        # convert Spacy PoS to Wordnet PoS so that it works easily with cosine_lesk
        return [(token.text, token.lemma_.lower(), self.wnTag(token.pos_)) for token in sentence]
            
    # convert Spacy tag to wordnet tag
    def wnTag(self, tag):
        convert_tag = {'NOUN':wn.NOUN, 'ADJ':wn.ADJ,
                      'VERB':wn.VERB, 'ADV':wn.ADV}
        try:
            return convert_tag[tag]
        except:
            return ''
                
    # select synset for each word ordered by likelihood
    # out of the box word sense dysambiguation algorithms were performing poorly
    def wsdSynSet(self, word_groups, sentence):
        output = []
        for token_group in word_groups:
            token, lemma, pos = token_group[0], token_group[1], token_group[2]
            if lemma in STOP_WORDS:
                synset = None
                output.append((token, lemma, synset))
                continue 
            try:
                synset = []
                for syn in wn.synsets(lemma, pos):
                    synset += syn.lemma_names()
                
                synset = list(OrderedDict.fromkeys(synset))
                synset.remove(lemma)
                if synset == []:
                    synset = None
            except:
                synset = None
                
            output.append((token, lemma, synset))
        
        return output
    
    def replace_synonyms(self, word_group, sentence):
        
        """
        utilize two geometric distributions as decribed in Zhang, Zhao, and Lecun 2015 "Char levell ConvNet for Text"
        first geo distribution determines how many words to replace in the sentence
            such that we are more likely to replace few words than a lot 
        
        next we identify which words that have synsets to replace randomly 
        
        second geometric distribution selects the index of the word in the synset to switch with 
            this way, the more similar the synonyms have a higher likelihood of being selected (need to order synonyms by likelihood)
            
        Utilize BERT language model to select best fit sentence (i.e., with lowest perplexity)
        - BERT is not deterministic so there is an element of randomness to best sentence selection
        - this is due to how it masks words for bidirectional computations
        I average three perplexity scores to help negate this effect
        """
        perp_best = 100000000000
        
        sentence_orig = sentence
        global best_word, synonyms, syn, replaced_idx, perp_tmp, best_sentence, best_group, best_poten
        
        for i in range(self.n_comparisons):
            sentence = sentence_orig # init
            # come up with number of replacements
            num_replacements = np.random.geometric(.5) 
            # indices to replace
            potential_indices = [i for i, group in enumerate(word_group) if group[2] is not None and group[2] is not []]
            to_replace = random.sample(potential_indices, k = min(num_replacements, len(potential_indices)))
            
            # replace with synonyms 
            for idx in to_replace:
                synonyms = word_group[idx][2]
                word = word_group[idx][0]
                
                # second geometric distribution to select which synonym to replace with
                syn_replace = min(np.random.geometric(.5)-1, len(synonyms)-1)
                try: 
                    syn = re.sub("_", " ", synonyms[syn_replace]) # wn using _ instead of space
                except:
                    syn = synonyms[syn_replace]
                    
                    
                sentence = re.sub(word, syn, sentence)
                
            # select best perplexity using BERT Language model
            perp_tmp = [self.lm.get_score(sentence) for i in range(3)]
            perp_tmp = sum(perp_tmp)/len(perp_tmp)
            if perp_tmp < perp_best:
                perp_best = perp_tmp
                best_sentence = sentence
        
            
        return best_sentence