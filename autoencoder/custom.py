import sys
sys.path.append('/home/abby')

import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import numpy as np 
import random
import autoencoder.common as common

# fully connected layers
# 2-3 encoding and decoding layers
# activation: 
    # encoding layers = sigmoid/tanh/log
    # decoding layers = relu
    
# Stacked denoising autoencoder
# loss = MSE + contractive loss + sparsity

# TODO: 
    # LSTM for sequential input
    # activation function - relu, elu, tanh, sigmoid?
    
activation = tf.nn.selu
def kl_divergence(rho, rho_hat):
    rho_hat = tf.cond(rho_hat <= 0, lambda: 1e-8, lambda: rho_hat)
    return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

class CustomAutoencoder():

    def __init__(self, num_input, reduce_dim=5, num_hid1=64, num_hid2=32, 
                 normalize=False, printfn=print):
        
        tf.reset_default_graph()
        
        self.print = printfn
        
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.data = []
            self.test_data = []
            self.normalize = normalize

            self.lam = 1e-4
            self.beta = 3
            self.rho = 0.05

            self.num_input = num_input
            self.num_hid1 = num_hid1
            self.num_hid2 = num_hid2
            self.reduce_dim = reduce_dim
            
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False)) 

            self.X1 = tf.placeholder('float', [None, self.num_input])
            self.X2 = tf.placeholder('float', [None, self.num_hid1])
            self.X3 = tf.placeholder('float', [None, self.num_hid2])
            self.noise1 = tf.placeholder('float', [None, self.num_input])
            self.noise2 = tf.placeholder('float', [None, self.num_hid1])
            self.noise3 = tf.placeholder('float', [None, self.num_hid2])
            
            self.Xhat1 = tf.placeholder('float', [None, self.reduce_dim])
            self.Xhat2 = tf.placeholder('float', [None, self.num_hid2])
            self.Xhat3 = tf.placeholder('float', [None, self.num_hid1])

            self.encoding_1 = tf.layers.dense(self.X1 + self.noise1, self.num_hid1, name='enc_fullc1', activation=activation)
            self.encoding_2 = tf.layers.dense(self.X2 + self.noise2, self.num_hid2, name='enc_fullc2', activation=activation)
            self.encoding_3 = tf.layers.dense(self.X3 + self.noise3, self.reduce_dim, name='enc_fullc3', activation=activation)

            self.decoding_1 = tf.layers.dense(self.Xhat1, self.num_hid2, name='dec_fullc1', activation=activation)
            self.decoding_2 = tf.layers.dense(self.Xhat2, self.num_hid1, name='dec_fullc2', activation=activation)
            self.decoding_3 = tf.layers.dense(self.Xhat3, self.num_input, name='dec_fullc3', activation=activation)

            self.loss1, self.opt1 = self.loss_fn(self.X1, self.Xhat3, self.decoding_3)
            self.loss2, self.opt2 = self.loss_fn(self.X2, self.Xhat2, self.decoding_2)
            self.loss3, self.opt3 = self.loss_fn(self.X3, self.Xhat1, self.decoding_1)
    
    def noise(self, size, sd=1.0):
        if self.normalize:
            sd = 0.10
        if not common.noisy:
            sd = 0.0
        return np.random.normal(size=size, loc=0.0, scale=sd)
    
    def loss_fn(self, y_true, encoded, y_pred):
        # TODO: get contractive part of loss
        
        # get sparse loss
        kl = 0.0
        eps = 1e-6
        if common.sparse:
            rho_hat = tf.reduce_mean(encoded) # WHAT IF THIS IS NEGATIVE? WHAT IF ITS 0
            kl = kl_divergence(self.rho, rho_hat)
        
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + self.beta * tf.reduce_sum(kl)
        optimizer = tf.train.AdadeltaOptimizer(common.learning_rate).minimize(loss) 
        return loss, optimizer

    def build_global_net(self):
        with self.graph.as_default():
            
            self.X = tf.placeholder('float', [None, self.num_input])
            self.Xnoise = tf.placeholder('float', [None, self.num_input])
            
            with tf.variable_scope('', reuse=True):
                e_weights1 = tf.get_variable("enc_fullc1/kernel").eval(self.sess) 
                e_weights2 = tf.get_variable("enc_fullc2/kernel").eval(self.sess) 
                e_weights3 = tf.get_variable("enc_fullc3/kernel").eval(self.sess)
                e_bias1 = tf.get_variable("enc_fullc1/bias").eval(self.sess) 
                e_bias2 = tf.get_variable("enc_fullc2/bias").eval(self.sess) 
                e_bias3 = tf.get_variable("enc_fullc3/bias").eval(self.sess) 

                d_weights1 = tf.get_variable("dec_fullc1/kernel").eval(self.sess) 
                d_weights2 = tf.get_variable("dec_fullc2/kernel").eval(self.sess) 
                d_weights3 = tf.get_variable("dec_fullc3/kernel").eval(self.sess) 
                d_bias1 = tf.get_variable("dec_fullc1/bias").eval(self.sess) 
                d_bias2 = tf.get_variable("dec_fullc2/bias").eval(self.sess) 
                d_bias3 = tf.get_variable("dec_fullc3/bias").eval(self.sess) 

            e1 = tf.layers.dense(self.X + self.Xnoise, self.num_hid1, name='net1', activation=activation, 
                                 kernel_initializer=tf.constant_initializer(e_weights1), 
                                 bias_initializer=tf.constant_initializer(e_bias1))
            e2 = tf.layers.dense(e1, self.num_hid2, name='net2', activation=activation, 
                                 kernel_initializer=tf.constant_initializer(e_weights2), 
                                 bias_initializer=tf.constant_initializer(e_bias2))
            self.encoder = tf.layers.dense(e2, self.reduce_dim, name='net3', activation=activation, 
                                           kernel_initializer=tf.constant_initializer(e_weights3), 
                                           bias_initializer=tf.constant_initializer(e_bias3))

            d1 = tf.layers.dense(self.encoder, self.num_hid2, name='net4', 
                                 activation=activation,
                                 kernel_initializer=tf.constant_initializer(d_weights1), 
                                 bias_initializer=tf.constant_initializer(d_bias1))
            d2 = tf.layers.dense(d1, self.num_hid1, name='net5', activation=activation, 
                                 kernel_initializer=tf.constant_initializer(d_weights2), 
                                 bias_initializer=tf.constant_initializer(d_bias2))
            self.decoder = tf.layers.dense(d2, self.num_input, name='net6', activation=activation,
                                          kernel_initializer=tf.constant_initializer(d_weights3), 
                                          bias_initializer=tf.constant_initializer(d_bias3))

            loss, optimizer = self.loss_fn(self.X, self.encoder, self.decoder)
            self.sess.run(tf.global_variables_initializer())
            
        return loss, optimizer
        
        
    def pretrain(self):
        
        batch_shape = (common.BATCH_SIZE,len(self.data[0]))
        
        # train first level: mapping from input_dim -> 20 -> input_dim
        for i in range(1, common.num_steps + 1):
            
            batch = common.random_sample(self.data, common.BATCH_SIZE)            
            enc1 = self.sess.run(self.encoding_1, feed_dict={self.X1: batch, self.noise1: self.noise(batch_shape)})             
            _, l = self.sess.run([self.opt1, self.loss1], feed_dict={self.X1: batch, self.Xhat3: enc1})
                
            if i % common.display_step == 0 or i == 1:
                self.print('Step %i: Minibatch Loss: %f' % (i, l))
                self.print('\t' + str(tf.reduce_mean(enc1).eval(session=self.sess)))

        # train second level: mapping from 20 -> 12 -> 20
        for i in range(1, common.num_steps + 1):
            
            batch = common.random_sample(self.data, common.BATCH_SIZE)
            enc1 = self.sess.run(self.encoding_1, feed_dict={self.X1: batch, self.noise1: np.zeros(batch_shape)})
            enc2 = self.sess.run(self.encoding_2, feed_dict={self.X2: enc1, self.noise2: self.noise(enc1.shape)})
            _, l = self.sess.run([self.opt2, self.loss2], feed_dict={self.X2: enc1, self.Xhat2: enc2})
                
            if i % common.display_step == 0 or i == 1:
                self.print('Step %i: Minibatch Loss: %f' % (i, l))
                self.print('\t' + str(tf.reduce_mean(enc2).eval(session=self.sess)))

        # train third level: mapping from 12 -> reduce_dim -> 12
        for i in range(1, common.num_steps + 1):
            batch = common.random_sample(self.data, common.BATCH_SIZE)

            enc1 = self.sess.run(self.encoding_1, feed_dict={self.X1: batch, self.noise1: np.zeros(batch_shape)})
            enc2 = self.sess.run(self.encoding_2, feed_dict={self.X2: enc1, self.noise2: np.zeros(enc1.shape)})
            enc3 = self.sess.run(self.encoding_3, feed_dict={self.X3: enc2, self.noise3: self.noise(enc2.shape)})
            
            _, l = self.sess.run([self.opt3, self.loss3], feed_dict={self.X3: enc2, self.Xhat1: enc3})
            
            if i % common.display_step == 0 or i == 1:
                self.print('Step %i: Minibatch Loss: %f' % (i, l))
                self.print('\t' + str(tf.reduce_mean(enc3).eval(session=self.sess)))
    
    def train(self, testiter=4):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.pretrain()
            
            batch_shape = (common.BATCH_SIZE, len(self.data[0]))

            # inititalize and train complete architecture
            loss, optimizer = self.build_global_net()
            steps = 3*common.num_steps
            for i in range(1, steps + 1):
                batch = common.random_sample(self.data, common.BATCH_SIZE)
                encoding = self.sess.run(self.encoder, feed_dict={self.X: batch, self.Xnoise: np.zeros(batch_shape)})
                _, l = self.sess.run([optimizer, loss], 
                                     feed_dict={self.X: batch, self.Xnoise: self.noise(batch_shape, sd=0.50)})

                if i % common.display_step == 0 or i == 1:
                    self.print('Step %i: Minibatch Loss: %f' % (i, l))
                    self.print('\t' + str(tf.reduce_mean(encoding).eval(session=self.sess)))
                    
    def test(self, iterations=4):
        test_batch_shape = (common.TEST_SIZE, len(self.test_data[0]))
        max_seen = np.ones(self.reduce_dim)
        with self.graph.as_default():
            for i in range(iterations):
                test_batch = common.random_sample(self.test_data, common.TEST_SIZE)
                pred_encoded = self.sess.run(self.encoder, feed_dict={self.X: test_batch, self.Xnoise: np.zeros(test_batch_shape)})
                pred = self.sess.run(self.decoder, feed_dict={self.X: test_batch, self.Xnoise: np.zeros(test_batch_shape)})
                common.log_test(test_batch, pred, self.print)
                
                for predicted in pred_encoded:
                    max_seen = np.maximum(max_seen, predicted)
       
        return max_seen
                
    # NOTE: you must call train() before you try to encode something.
    # Errors otherwise.
    def encode(self, x):
        x = np.array(x).reshape(1,29)
        encoded = self.sess.run(self.encoder, 
                                feed_dict={self.X: x, self.Xnoise: np.zeros(x.shape)})
        return encoded
    
    # TODO: test and train normalizations should use the same factors....
    def set_data(self, data):
        if self.normalize:
            data = common.normalize_obs(data)
        self.data = data
        
    def set_test_data(self, test_data):
        if self.normalize:
            test_data = common.normalize_obs(test_data)
        self.test_data = test_data