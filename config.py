# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:36:38 2019

@author: Administrator
"""

##各个参数定义
class config():
    max_epoch = 2
    #timestep_size = max_len=32
    keep_prob = 0.5
    batch_size = 32
    #layer_num = 2
    lr = 0.05
    input_size = embedding_size = 64
    class_num = 4
    hidden_size = 128
    model_save_path = 'bi-lstm.ckpt'
    shuffle = True
    clip_grad=1.25