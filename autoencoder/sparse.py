import tensorflow as tf
import numpy as np 
import random
from . import common

class SparseAutoencoder:

    def __init__(self, num_input, reduce_dim=5, normalize=False):
        self.data = []
        self.test_data = []
        self.normalize = normalize
        
        self.beta = 3
        self.rho = 0.05
        
        num_hidden_1 = 16
        num_hidden_2 = 8
        self.reduce_dim = reduce_dim
        self.num_input = num_input
        
        self.X = tf.placeholder('float', [None, self.num_input])
        
        weights = {
         'encoder_h1':tf.Variable(tf.random_normal([self.num_input, num_hidden_1])),
         'encoder_h2':tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
         'encoder_h3':tf.Variable(tf.random_normal([num_hidden_2, self.reduce_dim])),
                   
         'decoder_h1':tf.Variable(tf.random_normal([self.reduce_dim, num_hidden_2])),
         'decoder_h2':tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
         'decoder_h3':tf.Variable(tf.random_normal([num_hidden_1, self.num_input]))
        }
        
        biases = {
         'encoder_b1':tf.Variable(tf.random_normal([num_hidden_1])),
         'encoder_b2':tf.Variable(tf.random_normal([num_hidden_2])),
         'encoder_b3':tf.Variable(tf.random_normal([self.reduce_dim])),
                  
         'decoder_b1':tf.Variable(tf.random_normal([num_hidden_2])),
         'decoder_b2':tf.Variable(tf.random_normal([num_hidden_1])),
         'decoder_b3':tf.Variable(tf.random_normal([self.num_input]))
        }

        def encoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
            return layer_3

        def decoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
            return layer_3
        
        def kl_divergence(rho, rho_hat):
            return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

        self.encoder_op = encoder(self.X)
        self.decoder_op = decoder(self.encoder_op)
        
        y_pred = self.decoder_op
        y_true = self.X
        
        rho_hat = tf.reduce_mean(self.encoder_op, axis=0)
        kl = kl_divergence(self.rho, rho_hat)

        # Sparse autoencoding loss.
        self.loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + self.beta * tf.reduce_sum(kl)
        self.optimizer = tf.train.AdadeltaOptimizer(common.learning_rate).minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    def set_data(self, data):
        # TODO: normalize
        if self.normalize:
            data = common.normalize_obs(data)
        self.data = data
        
    def set_test_data(self, test_data):
        # TODO: normalize
        if self.normalize:
            test_data = common.normalize_obs(test_data)
        self.test_data = test_data

    def train(self):
        
        self.sess.run(tf.global_variables_initializer())

        print(len(self.data))
        for i in range(1, common.num_steps + 1):
            batch = random.sample(self.data, common.BATCH_SIZE)
            _, l = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: batch})
            if i % common.display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))

        print('--------')
        print(len(self.test_data))
        for i in range(4):
            test_batch = random.sample(self.test_data, common.TEST_SIZE)
            pred = self.sess.run(self.decoder_op, feed_dict={self.X: test_batch})
            common.log_test(test_batch, pred)
    

