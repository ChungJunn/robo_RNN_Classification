'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

class FSIterator:
    def __init__(self, filename, batch_size=4, just_epoch=False):
        self.batch_size = batch_size
        self.just_epoch = just_epoch
        self.fp = open(filename, 'r')

    def __iter__(self):
        return self

    def reset(self):
        self.fp.seek(0)

    def __next__(self):
        bat_seq = []
        bat_lbs = []

        end_of_data = 0
        for i in range(self.batch_size):
            seq = self.fp.readline()
            if seq == "":
                if self.just_epoch:
                    end_of_data = 1
                    if self.batch_size==1:
                        raise StopIteration
                    else:
                        break
                self.reset()
                seq = self.fp.readline()

            seq_f = [float(s) for s in seq.split(',')]
            seq_l = int(seq_f[-1])
            seq_f = seq_f[:-1]

            bat_seq.append(seq_f)
            bat_lbs.append(seq_l)
        
        bat_seq = np.array(bat_seq)
        bat_seq = trimBatch(bat_seq)
        
        mask = getMask(bat_seq) # TimeSteps BatchSize

        x_data = self.prepare_data(bat_seq)
        y_data = np.array(bat_lbs).reshape(1,-1)

        return x_data, y_data, mask, end_of_data

    def prepare_data(self, seq):
        seq = addPadding(seq) # zero padding
        
        x_data = addDelta(seq) # TimeSteps BatchSize InputDim 

        return x_data #y_data

def getSeq_len(row):
    '''
    returns: count of non-nans (integer)
    adopted from: M4rtni's answer in stackexchange
    '''
    return np.count_nonzero(~np.isnan(row))

def getMask(batch):
    '''
    returns: boolean array indicating whether nans
    '''
    return (~np.isnan(batch)).astype(np.int32).transpose()

def trimBatch(batch):
    '''
    args: npndarray of a batch (bsz, n_features)
    returns: trimmed npndarray of a batch.
    '''
    max_seq_len = 0
    for n in range(batch.shape[0]):
        max_seq_len = max(max_seq_len, getSeq_len(batch[n]))

    if max_seq_len == 0:
        print("error in trimBatch()")
        sys.exit(-1)

    batch = batch[:,:max_seq_len]
    return batch

def addPadding(data):
    '''
    args: 2D npndarray with nans
    returns npndarray with nans padded with 0's
    '''
    for n in range(data.shape[0]):
        for i in range(data.shape[1]):
            if np.isnan(data[n,i]): data[n,i] = 0
    return data

def getDelta(col_prev, col): # helper for addDelta()
    delta = (col - col_prev).reshape(-1,1)

    return np.hstack([col, delta])

def addDelta(batch):
    timesteps = []
    # treat first column
    delta = np.zeros((batch.shape[0],2))
    timesteps.append(delta)

    # tread the rest
    for i in range(1, batch.shape[1]):
        timesteps.append(getDelta(batch[:,i-1].reshape(-1,1), batch[:,i].reshape(-1,1)))

    samples = np.stack(timesteps)

    return samples

def batchify(data, bsz, labels):
    batches = []
    
    n_samples = data.shape[0]
    for n in range(0, n_samples, bsz):
        if n+bsz > n_samples: #discard remainder #TODO: use remainders somehow
            break
        batch = data[n:n+bsz]
        target = labels[n:n+bsz]

        batch = trimBatch(batch)
        mask = getMask(batch)
        
        batch = addPadding(batch)

        batch = addDelta(batch)
        mask = mask.transpose()

        batches.append([batch, mask, target])

    return batches

def prepareData():
    df_train = pd.read_csv("./data/classification_train.csv")
    df_valid = pd.read_csv("./data/classification_valid.csv")

    np_train = np.asarray(df_train)
    np_valid = np.asarray(df_valid)
    
    np_data = np_train[:,:-1]
    np_labels = np_train[:,-1].reshape(-1,1)
    
    np_vdata = np_valid[:,:-1]
    np_vlabels = np_valid[:,-1].reshape(-1,1)

    return np_data, np_labels, np_vdata, np_vlabels

if __name__ == "__main__":
    
    
    iterator = FSIterator("./data/classification_train.csv")

    for item in iterator:
        print(item)
