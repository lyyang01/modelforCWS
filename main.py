# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:47:03 2019

@author: Administrator
"""

#import tensorflow as tf
#import numpy as np
#import os, argparse, time, random
from BiLSTMmodel import bilstm_model
from data import read_corpus, read_dictionary, random_embedding
from config import config

## get char embeddings
word2id = read_dictionary('vocab')
##随机产生embedding
embeddings = random_embedding(word2id, config.embedding_size)

paths={'log_path':'logger//', 'model_path':'./model1/','result_path':'result//'}

#TODO 注意：model_path！！这是个坑啊！！

model = bilstm_model(embeddings, paths, word2id, config=config)
model.build_graph()


#train_data = read_corpus('pku_training.utf8')
test_data = read_corpus('pku_test_gold.utf8')

## train model on the whole training data
#print("train data: {}".format(len(train_data)))
#model.train(train_data=train_data) 

print("test data: {}".format(len(test_data)))
model.test(test_data=test_data) 

