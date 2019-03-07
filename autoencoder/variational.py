import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import numpy as np 
import random
from . import common

class VariationalAutoencoder():
    
    def __init__(self, num_input, reduce_dim=5, normalize=False):
        
        self.graph = tf.Graph()
        with self.graph.as_default():
        
            self.data = []
            self.test_data = []
            self.normalize = normalize

            self.num_input = num_input
            num_hid1 = 12
            num_hid2 = 8
            self.reduce_dim = reduce_dim

            self.X = tf.placeholder('float', [None, self.num_input])

            ## Encoder ##
            f1 = fc(self.X, num_hid1, scope='enc_fullc1', activation_fn = tf.nn.elu)
            f2 = fc(f1, num_hid2, scope='enc_fullc2', activation_fn = tf.nn.elu)

            self.z_mu = fc(f2, self.reduce_dim, scope='enc_mu', activation_fn=None)
            self.log_sigma_z_sq = fc(f2, self.reduce_dim, scope='enc_sigma_sq', activation_fn=None)

            # Generate z from the normal distribution
            eps = tf.random_normal(shape=tf.shape(self.log_sigma_z_sq), mean=0, stddev=1, dtype=tf.float32)
            self.z = self.z_mu + tf.sqrt(tf.exp(self.log_sigma_z_sq)) * eps

            ## decoder ##
            d1 = fc(self.z, num_hid2, scope='dec_fc1', activation_fn=tf.nn.elu)
            d2 = fc(self.z, num_hid1, scope='dec_fc2', activation_fn=tf.nn.elu)
            self.x_hat = fc(d2, self.num_input, scope='dec_out', activation_fn=tf.sigmoid)

            y_pred = self.x_hat
            y_true = self.X

            # latent distribution loss
            latent_loss = -0.5 * tf.reduce_sum(
                1 + self.log_sigma_z_sq - tf.square(self.z_mu) - tf.exp(self.log_sigma_z_sq),
                axis=1
            )

            self.loss = tf.reduce_mean(tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + latent_loss)
            self.optimizer = tf.train.AdamOptimizer(common.learning_rate).minimize(self.loss)
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
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            for i in range(1, common.num_steps + 1):
                batch = random.sample(self.data, common.BATCH_SIZE)
                _, l = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: batch})
                if i % common.display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, l))

            print('--------')
            print(len(self.test_data))
            for i in range(4):
                test_batch = random.sample(self.test_data, common.TEST_SIZE)
                pred = self.sess.run(self.x_hat, feed_dict={self.X: test_batch})
                common.log_test(test_batch, pred)
