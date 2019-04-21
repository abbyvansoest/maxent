import os
import joblib
import tensorflow as tf
import random

import cvxpy as cvx
import numpy as np

def restore_tf_graph(sess, fpath):

    tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                fpath
            )

    model_info = joblib.load(os.path.join(fpath, 'model_info.pkl'))
    graph = tf.get_default_graph()
    model = dict()
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
    return model

def load_policy(fpath, itr='last'):

    # handle which epoch to load from
    saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
    itr = '%d'%max(saves) if len(saves) > 0 else ''

    # load the things!
    graph = tf.Graph() # each policy needs unique graph
    with graph.as_default():
        sess = tf.Session()
        model = restore_tf_graph(sess, os.path.join(fpath, 'simple_save'+itr))

        # make function for producing an action given a single state
        action_op = model['pi']
        get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(os.path.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def proj_unit_simplex(y):
    '''
    Returns the point in the simplex a^Tx = 1, x&amp;amp;amp;amp;gt;=0 that is
     closest to y (according to Euclidian distance)
    '''
    d = len(y)
    a = np.ones(d)
    # setup the objective and constraints and solve the problem
    x = cvx.Variable(d)
    obj = cvx.Minimize(cvx.sum_squares(x - y))
    constr = [x >= 0, a*x == 1]
    prob = cvx.Problem(obj, constr)
    prob.solve()
 
    return np.array(x.value)

def geometric_weights(N):
    weights = [.90**(N-i) for i in range(N)]
    weights = proj_unit_simplex(weights)
    weights = np.absolute(weights) / weights.sum()
    return weights

# TODO(abbyvs): need to select actions according to final learned weights.
# need to save weights from training and load here.
class Explorer:

    def __init__(self, expert_dir, env_fn):
        # Load all policies
        self.env = env_fn()
        self.get_actions = []
        idx = 0
        for model_dir in sorted(os.listdir(expert_dir)):
            cur_dir = expert_dir+model_dir
            print('-----------------')
            print(cur_dir)
            if not os.path.isdir(cur_dir):
                continue
            if 'simple_save' not in os.listdir(cur_dir):
                self.get_actions.append('null get_action op')
                continue

            _, get_action = load_policy(cur_dir)
            self.get_actions.append(get_action)
        print('Total policies = %d' % len(self.get_actions))
        
        # Temporary weighting fix: select actions according to geometric weighting.
        # get_actions has all action ops from 
        self.weights = geometric_weights(len(self.get_actions))

    def sample_action(self, obs):
        
        indexes = np.arange(len(self.get_actions))
        idx = np.random.choice(indexes, p=self.weights)
        
        if idx == 0:
            action = self.env.action_space.sample()
        else:
            action = self.get_actions[idx](obs)
        return action

    def test(self, steps=1000):
        o = self.env.reset()
        o = self.env.env.state_vector()
        for i  in range(steps):
            a = self.sample_action(o)
            o, r, d, _ = self.env.step(a)
            o = self.env.env.state_vector()




