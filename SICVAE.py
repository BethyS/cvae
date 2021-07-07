# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:47:00 2021

@author: Bethe
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Sum 

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import utils
## ---------------------------------------------------------------
class CVAE:
    def __init__(self,
                 dista_weight,
                 spect_res,
                 temp__res,
                 args
                  ):
        super(CVAE, self).__init__()
        self.dista_weight = dista_weight
        self.spect_res = spect_res # dataset height 
        self.temp__res = temp__res # dataset width
        self.batch_size = arg.batch_size
        self.latent_dim = arg.latent_dim
        self.lr = tf.Variable(args.lr, trainable=False)
        self.c_i = 0
        self.encoder = tf.keras.Sequential(
            [
                Input(input_shape=(spect_res, temp__res, 1)),
                Conv2D(filters=32, kernel_size=3, padding = 'same', activation='relu'),
                Conv2D(filters=64, kernel_size=3, padding = 'same', activation='relu'),
                MaxPooling2D(),
                Conv2D(filters=64, kernel_size=3, padding = 'same', activation='relu'),
                MaxPooling2D(),
                Conv2D(filters=64, kernel_size=3, padding = 'same', activation='relu'),
                Flatten()
                Dense(latent_dim + latent_dim)
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Input(input_shape=(latent_dim,)),
                Dense(units=1*11*32, activation='relu'),
                Reshape(target_shape=(1, 11, 32)),
                UpSampling2D();
                Conv2DTranspose(filters=64, kernel_size=3, padding='same',
                    activation='relu'),
                UpSampling2D(size=(2,2)),
                Conv2DTranspose(filters=64, kernel_size=(3,5),strides=1,activation='relu'),
                Conv2DTranspose(filters=32, kernel_size=(3,3),strides=1,activation='relu'),
                Conv2DTranspose(filters=64, kernel_size=(3,4),strides=1,activation='relu'),
                UpSampling2D(size=(1,2)),
                UpSampling2D(size=(1,2))
                Conv2DTranspose(filters=1, kernel_size=3, padding='same',
                    activation='relu')
            ]
        )
    def encode(self, x):
        mean=self.Dense(x,units=latent_dim, activation='relu')
        logstddev=Dense(x,units=latent_dim,activation='exp')
        latent_c= Dense(x,units=latent_dim, activation='relu')
        return mean, logstddev,latent_c
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def reparameterize(self, mean, logstddev,latent_c):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logstddev * .5) + mean
    
    def decode(self, z):
        output = self.decoder(z)
        return output
    def Custom_loss(self,):
        
    
    
