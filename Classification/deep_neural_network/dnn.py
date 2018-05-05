# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:43:38 2018

@author: shifuddin
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("E:\ML\Classification\deep_neural_network\data", one_hot=True)

n_node_h1 = 500
n_node_h2 = 500
n_node_h3 = 500

n_classes = 10
batch_size = 100

X = tf.placeholder('float', [None, 786])
Y = tf.placeholder('float')

def neural_network_model(data):
    hidden_l_1 = {'weights':tf.variable(tf.random_normal([786, n_node_h1])), 'biases':tf.variable(tf.random_normal(n_node_h1))}
    
