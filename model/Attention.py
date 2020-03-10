# encoding: utf-8

import tensorflow as tf
from enum import Enum


class SentenceMode(Enum):
    ATTENTION = 0
    FINAL_STATE = 1

class Attention(object):
    def __init__(self,input,attention_size,time_major=False,return_alphas=False):
        self.inputs = input
        self.attention_size = attention_size
        self.time_major = time_major
        self.return_alphas = return_alphas
        self.seed = 317

    def attentionModel(self):
        if isinstance(self.inputs,tuple):
            self.inputs = tf.concat(self.inputs,2)
        if self.time_major:
            self.inputs = tf.transpose(self.inputs,[0,2,1]) # (T,B,D) =》 (B,T,D)

        self.hidden_size = self.inputs.shape[2].value # D value-hidden sixe of the RNN layer
        self.W = tf.Variable(tf.random_normal([self.hidden_size,self.attention_size],stddev=0.1,seed=self.seed))
        self.b = tf.Variable(tf.random_normal([self.attention_size],stddev=0.1,seed=self.seed))
        self.u = tf.Variable(tf.random_normal([self.attention_size],stddev=0.1,seed=self.seed))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestep
            # the shape of 'v' is (B,T,D)*(D,A) = (B,T,A),where A=attention_size
            self.v = tf.tanh(tf.tensordot(self.inputs,self.W,axes=1)+self.b)
        self.uv = tf.tensordot(self.v,self.u,axes=1)
        self.alphas = tf.nn.softmax(self.uv,name='alphas') # (B,T)shape

        # Output of (Bi-)RNN is reduced with attention vector;
        self.output = tf.reduce_sum(self.inputs*tf.expand_dims(self.alphas,-1),-1)

        if not self.return_alphas:
            return self.output
        else:
            return self.output,self.alphas