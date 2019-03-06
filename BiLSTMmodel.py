# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:19:46 2019

@author: liuyang
"""

import tensorflow as tf
#from config import config
from tensorflow.contrib import rnn
import time,sys
from data import  batch_yield,pad_sequences,tag2label
from tensorflow.contrib.rnn import LSTMCell
import os
from eval import conlleval
import logging
'''
  For Chinese word segmentation
'''

#模型定义
class bilstm_model():
    def __init__(self,embeddings, paths, vocab ,config):
        self.config=config
        self.embeddings=embeddings
        self.model_path = paths['model_path']
        self.tag2label=tag2label
        self.vocab = vocab
        self.update_embedding=True
        self.logger = self.get_logger(paths['log_path'])
        self.result_path = paths['result_path']


    def print_op(self):
        self.a_print = tf.Print(self.logits,['logit_shape:',self.logits])
        return self.a_print
       
        
        
        
    def build_graph(self):
        self.add_placeholders()
        self.bi_lstm_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()
        self.print_op()
    
    ##需要使用feed_dict  feed到模型中的真实的数据,由于还没传入，所以先用占位符定义
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])
        self.labels=tf.placeholder(tf.int32, [None, None])
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        #self.a_print=tf.placeholder(tf.int32, [None, None])
        
    #def weight_variable(shape):
    #    initial = tf.truncated_normal(shape, stddev=0.1)
    #    return tf.Variable(initial)
    
    #def bias_variable(shape):
    #    initial = tf.constant(0.1, shape=shape)
    #    return tf.Variable(initial)
    
    
    def bi_lstm_op(self):
        """build the biLSTMs network. Return the y_pred"""
        with tf.variable_scope("bi-lstm"):
            _word_embeddings = tf.Variable(self.embeddings,
                                               dtype=tf.float32,
                                               trainable=self.update_embedding, #是否在训练过程中更新该变量
                                               name="_word_embeddings")
            self.word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                         ids=self.word_ids,
                                                         name="word_embeddings")
            #inputs = tf.nn.embedding_lookup(embedding, self.x)
            #双向LSTM层
            cell_fw = LSTMCell(self.config.hidden_size) #前向
            cell_bw = LSTMCell(self.config.hidden_size) #后向
            #shape (batchsize, timestep, hidden_size)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=self.word_embeddings,
                    sequence_length=self.sequence_lengths,
                    dtype=tf.float32)
            #前向后向concat到一起获得最终输出
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)  #最终输出
        
        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.config.hidden_size, self.config.class_num],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.config.class_num],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.config.hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.config.class_num])
            temp=tf.reshape(tf.argmax(self.logits, -1),[-1])
            correct_prediction = tf.equal(tf.cast(temp, tf.int32), tf.reshape(self.labels, [-1]))


            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #self.a_print = tf.Print(self.logits,['logit_shape:',self.logits])

            #print(1)
    
    def loss_op(self):      
      
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.cost = tf.reduce_mean(losses)
        self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
        self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

#生成优化器并进行梯度下降计算
    def trainstep_op(self):       
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)

        grads_and_vars = optimizer.compute_gradients(self.cost)
        grads_and_vars_clip = [[tf.clip_by_value(g, -self.config.clip_grad, self.config.clip_grad), v] for g, v in grads_and_vars] ##梯度截断
        self.train_op = optimizer.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            
    

###以上均为模型的定义。#########################################################################################
     ##初始化参数
    def init_op(self):
        self.init_op = tf.global_variables_initializer()
        
    #def print_op(self):
    #    self.print_op=tf.Print(self.num_batches,[self.num_batches,1])

    def run_one_epoch(self, sess, train_data, epoch, saver):
        num_batches = (len(train_data) + self.config.batch_size - 1) // self.config.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        

        ##对训练数据生成batches
        batches = batch_yield(train_data, self.config.batch_size, self.vocab, shuffle=self.config.shuffle)
        #tf.Print(11)
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            sys.stdout.write('\n')
            step_num = epoch * num_batches + step + 1
            #tf.Print(step_num)

            feed_dict, _ = self.get_feed_dict(seqs, labels=labels)
           
            _, loss_train, step_num_ ,accuracy= sess.run([self.train_op,  self.cost, self.global_step, self.accuracy],
                                                         feed_dict=feed_dict)
            
            ####显示显示
            if step + 1 == 1 or (step + 1) % 30 == 0 or step + 1 == num_batches:
                sys.stdout.write(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {} acc: {:.4}'.format(start_time, epoch + 1, step + 1,
                                                                         loss_train, step_num, accuracy))
                
                sys.stdout.write('\n')

            ##这里需要保存模型
            
            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)        
            
            
    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
       
        '''生成三个数据，wordid, sequence_lengths, labels'''
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        
        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    
    def train(self, train_data):
        """
          开始训练
        """
        saver = tf.train.Saver(tf.global_variables())
        ##开始训练，运行一个session
        with tf.Session() as sess:
            sess.run(self.init_op)
            for epoch in range(self.config.max_epoch):
                self.run_one_epoch(sess, train_data, epoch, saver)
    
##测试      
    def test(self, test_data):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info('=========== testing ===========')
            model_file=tf.train.latest_checkpoint(self.model_path)
            #model_file=os.path.join('D://modelforCWS//-596')
            saver.restore(sess, model_file)  ##读入模型
            label_list, seq_len_list = self.dev_one_epoch(sess, test_data)
          
            self.evaluate(label_list, seq_len_list, test_data)
    
    def dev_one_epoch(self, sess, dev):
      
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.config.batch_size, self.vocab, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list
    
    def predict_one_batch(self, sess, seqs):
  
        feed_dict, seq_len_list = self.get_feed_dict(seqs)
        
        label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
        return label_list, seq_len_list
    
    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        
        #label2tag = {}
        #for tag, label in self.tag2label.items():
        #    label2tag[label] = tag if label != 0 else label
        
        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            
            sent_res = []
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], label_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)
    
    def get_logger(self, filename):
        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        return logger