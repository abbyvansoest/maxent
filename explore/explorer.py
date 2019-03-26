import os
import joblib
import tensorflow as tf
import random

def restore_tf_graph(sess, fpath):

    tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                fpath
            )

    model_info = joblib.load(os.path.join(fpath, 'model_info.pkl'))
    print(model_info)

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

class Explorer:

    def __init__(self, expert_dir, env_fn):
        # Load all policies
        # TODO(abbyvs): amke sure sorted by epoch
        self.env = env_fn()
        self.get_actions = []
        idx = 0
        for model_dir in sorted(os.listdir(expert_dir)):
            cur_dir = expert_dir+model_dir
            print('-----------------')
            print(cur_dir)
            if 'simple_save' not in os.listdir(cur_dir):
                continue

            _, get_action = load_policy(cur_dir)
            self.get_actions.append(get_action)

    def sample_action(self, obs):
        idx = random.randint(0, len(self.get_actions)-1)
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




