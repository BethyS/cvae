import numpy as np
import tensorflow as tf
from tensorflow import layers


class CVAE():
    def __init__(self,
                 dista_weight,
                 spect_res,
                 temp__res,
                 args
                  ):
        self.dista_weight = dista_weight
        self.spect_res = spect_res # dataset height 
        self.temp__res = temp__res # dataset width
        self.batch_size = arg.batch_size
        self.latent_size = arg.latent_size
        self.lr = tf.Variable(args.lr, trainable=False)
        self.stddev = args.stddev
        self.mean = args.mean
        self.c_i = tf.Variable(args., trainable=False)
        # c_i - optional cluster_vectors, can be specified separately

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_dim = 10         # dimension of condition-vector (label)
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError
        self.c_i = None #
        self.c_i_ph = None
        
    def conditional_Vae(self, X, Y, keep_prob):

        with tf.variable_scope("conditional", reuse = tf.AUTO_REUSE):
            X_input = tf.concat((X,Y), axis =1)
            
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)
            #network
            enc = lrelu(conv2d(X_input, 32, 3, 3, name='en_conv1',in))
            enc = lrelu(drop_out(conv2d(enc, 64, 3, 3, name='en_conv2'), is_training=is_training),keep_prob)
            enc = lrelu(conv2d(enc, 64, 3, 3, name='en_conv3'), is_training=is_training)
            enc= maxpl
            
            net = drop_out(leaky(dense(X_input, self.n_hidden[0], name = "Dense_1")), keep_prob)
            net = drop_out(leaky(dense(net, self.n_hidden[1], name="Dense_2")), keep_prob)
            net = dense(net, self.n_z*2, name ="Dense_3")
            mean = net[:,:self.n_z]
            std = tf.nn.softplus(net[:,self.n_z:]) + 1e-6

        return mean, std
        