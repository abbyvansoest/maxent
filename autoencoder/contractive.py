import tensorflow as tf
import numpy as np 
import random
from . import common

class ContractiveAutoencoder():

    def __init__(self, num_input, reduce_dim=5, normalize=False):
                
        self.data = []
        self.test_data = []
        self.normalize = normalize
        
        lam = 1e-3
        
        self.num_input = num_input
        num_hid1 = 16
        self.reduce_dim = reduce_dim
        
        self.X = tf.placeholder('float', [None, self.num_input])
        
        weights = {
         'encoder_h1':tf.Variable(tf.random_normal([self.num_input, num_hid1])),
         'encoder_h2':tf.Variable(tf.random_normal([num_hid1, self.reduce_dim])),
         'decoder_h1':tf.Variable(tf.random_normal([self.reduce_dim, num_hid1])),
         'decoder_h2':tf.Variable(tf.random_normal([num_hid1, self.num_input]))
        }
        
        biases = {
         'encoder_b1':tf.Variable(tf.random_normal([num_hid1])),
         'encoder_b2':tf.Variable(tf.random_normal([self.reduce_dim])),
         'decoder_b1':tf.Variable(tf.random_normal([num_hid1])),
         'decoder_b2':tf.Variable(tf.random_normal([self.num_input]))
        }

        def encoder(x):
            encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, weights['encoder_h2']), biases['encoder_b2']))
            return encoder_layer_1, encoder_layer_2

        def decoder(x):  
            decoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
            decoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_layer_1, weights['decoder_h2']), biases['decoder_b2']))
            return decoder_layer_1, decoder_layer_2

        el1, self.encoder_op = encoder(self.X)
        dl1, self.decoder_op = decoder(self.encoder_op)
        
        y_pred = self.decoder_op
        y_true = self.X
        
        # get contractive part of loss
        dh1 = el1 * (1 - el1)
        dh2 = self.encoder_op * (1 - self.encoder_op)
        contractive = lam * (tf.reduce_sum(dh1**2 * tf.reduce_sum(weights['encoder_h1']**2)) \
                             + tf.reduce_sum(dh2**2 * tf.reduce_sum(weights['encoder_h2']**2)))

        # contractive autoencoding loss.
        self.loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + contractive
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
        