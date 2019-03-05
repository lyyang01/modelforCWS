# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:02:23 2019

@author: Administrator
"""

import os


eval_perl = "./conlleval_rev.pl"
label_path='result//output.txt'
metric_path='result//test.txt'
os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))