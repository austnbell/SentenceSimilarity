# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 12:06:44 2019

Core script for Siamese networks for text similarity
Inspired by: "Siamese Recurrent Architectures for Learning Sentence Similarity" by Mueller and Thyagarajan

@author: Austin Bell
"""
basePath = "C:/Users/Austin Bell/Documents/SemSearch"
    
import sys
sys.path.append(basePath)
#from AugmentData import *
from torch.nn.utils import clip_grad_norm_
import torch
from torch.autograd import Variable
from Models import *
from Utils import *
from FastText_vocab import *
from parameters import *
from AugmentData import *
from gensim.models import FastText, KeyedVectors
from scipy.stats.stats import pearsonr 

params = parameters(basePath)
#ftModel = FastText.load_fasttext_format(params.ftPath)
print("Loaded Fasttext")
        

###################################################################
## Source and augment data
###################################################################
# the Sick dataset utilizes an augmented synonym training set whereas the STS pretraining data does not

if params.data == "SICK":
    try: 
        train = open(params.trainAugFile, "r", encoding = "utf-8").readlines()
        print("Loaded Train")
    
    except:
        augmentor = dataAugmentor(params.inFile, params.trainAugFile, params.augFile)
        augmentor.augment()
        train = open(params.outFile, "r", encoding = "utf-8").readlines()
        
elif params.data == "STS":
    train = open(params.inFile, "r", encoding = "utf-8").readlines()

# both datasets use a validation set
val = open(params.testFile, "r", encoding = "utf-8").readlines()

###################################################################
## Set up data
###################################################################

# if pretraining is set to false then this class handles the vocabulary and embedding extension internally
vocab = FTvocab(ftModel, train + val)
if params.pretrain == False:
    vocab.load(basePath + "/Data/Models/STS vocab.txt", basePath + "/Data/Models/STS embed.npy")
vocab.getWords()
vocab.createEmbeddings()
vocab_sz = len(vocab)
embed_dim = 300
vocab.save(params.vocab_path, params.embed_path)
print("Created Vocab")

sMapTrain = map2idx(train, params, vocab)
sMapVal = map2idx(val, params, vocab)

train_inputs = inputTensors(sMapTrain)
train_loader = data.DataLoader(Dataset(*train_inputs), batch_size=params.bs, shuffle = True)

val_inputs = inputTensors(sMapVal)
val_loader = data.DataLoader(Dataset(*val_inputs), batch_size =params.bs) 


print("Prepped data and preparing model...")
###################################################################
## Modelling 
###################################################################
lr = params.lr

Model = SiameseFTnetwork(vocab_sz, embed_dim, torch.FloatTensor(vocab.vectors), params).to(params.device)
optimizer = torch.optim.Adam(Model.parameters(), lr = lr)

criterion = nn.MSELoss().to(params.device)

num_batches = len(train_loader)
epoch_no_improvement = 0
reduce_lr = 0
best_val_loss = 100

for i in range(params.epochs):
    Model.train()
    total_loss = 0
    train_losses = []
    train_scores = []
    train_preds = []
    
    for j, (padded_sent_a, padded_sent_b, senta_lengths, sentb_lengths, gs) in enumerate(train_loader):
        #print(j)
        padded_sent_a = Variable(padded_sent_a).to(params.device)
        padded_sent_b = Variable(padded_sent_b).to(params.device)
        gs = Variable(torch.exp(-gs)).to(params.device)
        
        
        Model.zero_grad()
        
        L1_loss = Model(padded_sent_a, padded_sent_b)
        
        # backward
        loss = criterion(L1_loss, gs)
        loss.backward()
        
        # Clip gradients
        clip_grad_norm_(Model.LSTM_a.parameters(), .25)
        clip_grad_norm_(Model.LSTM_b.parameters(), .25)
        
        optimizer.step()
        
        # tabulate
        #train_losses += [-np.log(loss.cpu().detach().numpy())]
        train_losses += [loss.cpu().detach().numpy()]
        train_preds += list(L1_loss.cpu().detach().numpy())
        train_scores += list(gs.cpu().detach().numpy())
        
    # print training information 
    avg_train_mse = sum(train_losses) / len(train_losses)
    print("Epoch: {} out of: {}".format(i+1, params.epochs))
    print("Batch number: {} out of: {}".format(j, num_batches +1))
    print("MSE: {}".format(avg_train_mse))
    print("Pearson: {}".format(pearsonr(train_preds, train_scores)))
    
    # test early stop and incorporate learning rate annealing
    # by running on validation set
    if i >= 4:
        val_losses = []
        for j, (padded_sent_a, padded_sent_b, senta_lengths, sentb_lengths, gs) in enumerate(val_loader):
            padded_sent_a = Variable(padded_sent_a).to(params.device)
            padded_sent_b = Variable(padded_sent_b).to(params.device)
            gs_val = Variable(torch.exp(-gs)).to(params.device)
            
            L1_loss = Model(padded_sent_a, padded_sent_b)
            loss = criterion(L1_loss, gs_val)
            val_losses += [loss.cpu().detach().numpy()]
            
        # has the model improved?
        avg_val_loss = sum(val_losses) / len(val_losses)
        print("Average Validation Loss: {}".format(avg_val_loss))
        if avg_val_loss <= best_val_loss:
            epoch_no_improvement = 0
            best_val_loss = avg_val_loss
            best_model = Model
            
            # save the improved iteration
            torch.save(Model.state_dict(), params.model_path)
                
        else:
            epoch_no_improvement += 1
            reduce_lr += 1
            print("No improvement {}".format(epoch_no_improvement))
            
    # reduce learning rate if no improvement for long enough
    if reduce_lr == params.reduce_lr_epochs:
        lr = lr*params.annealing_wgt
        reduce_lr = 0
        print("New LR {}".format(lr))
        # update
        for p in optimizer.param_groups:
            p['lr'] = lr
        
        
    # stop the network if we there are n epochs without any improvement       
    if epoch_no_improvement == params.early_stop:
        final_epoch = j
        print("Best Validation score: {}".format(best_val_loss))
        break
    
    



    
# evaluate
val_losses = []
val_preds = []
val_scores = []
for j, (padded_sent_a, padded_sent_b, senta_lengths, sentb_lengths, gs) in enumerate(train_loader):
    padded_sent_a = Variable(padded_sent_a).to(params.device)
    padded_sent_b = Variable(padded_sent_b).to(params.device)
    gs = Variable(torch.exp(-gs)).to(params.device)
       
    L1_loss = best_model(padded_sent_a, padded_sent_b)
    loss = criterion(L1_loss, gs)
    
    val_losses += [loss.cpu().detach().numpy()]            
    val_preds += list(L1_loss.cpu().detach().numpy())
    val_scores += list(gs.cpu().detach().numpy())

avg_val_mse = sum(val_losses) / len(val_losses)
print("MSE: {}".format(avg_val_mse))    
print("Pearson: {}".format(pearsonr(val_preds, val_scores)))
  

