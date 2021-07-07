## ## Bethelhem S.: CVAE- 1D
## ## (Version 1.0.1, built: 2020-10-01)
## ## Copyright (C)2020 Bethelhem SEifu

#import library
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
# Load Dataset and preprocessing
sensor_id = [113, 196, 201, 230, 323, 358, 363, 378, 389, 395, 397, 62]
rawdata = []
# model 
#encoder 
class CVAE():
    def __init__(self,
                 vocab_size,
                 args
                  ):

        self.vocab_size = vocab_size
        self.batch_size = args.batch_size
        self.latent_size = args.latent_size
        self.lr = tf.Variable(args.lr, trainable=False)
        self.num_prop = args.num_prop
        self.stddev = args.stddev
        self.mean = args.mean
        self.unit_size = args.unit_size
        self.n_rnn_layer = args.n_rnn_layer
        
        self._create_network()
    


