# Code derived from: https://spinningup.openai.com/en/latest/algorithms/sac.html

import numpy as np
import tensorflow as tf
import core
from core import get_vars
from spinup.utils.logx import EpochLogger

import reacher_utils

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)        
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class ReacherSoftActorCritic:

    def var_scope(self, var):
        return self.main_scope + '/' + var

    def __init__(self, env_fn, reward_fn=[], actor_critic=core.mlp_actor_critic, xid=0, seed=0, max_ep_len=1000,
        gamma=.99, alpha=0.2, lr=1e-3, polyak=0.995, replay_size=int(1e6), 
        ac_kwargs=dict(), logger_kwargs=dict(), normalization_factors=[], learn_reduced=False):
        
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.main_scope = 'main' + str(xid)
        self.target_scope = 'target' + str(xid)

        self.logger = EpochLogger(**logger_kwargs)

        self.max_ep_len = max_ep_len
        self.reward_fn = reward_fn
        self.normalization_factors = normalization_factors
        self.learn_reduced = learn_reduced
        
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = reacher_utils.state_dim
        if self.learn_reduced:
            self.obs_dim = reacher_utils.expected_state_dim
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = self.env.action_space

        self.graph = tf.Graph()
        with self.graph.as_default():
             # Inputs to computation graph
            self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(self.obs_dim, self.act_dim, self.obs_dim, None, None)

            # Main outputs from computation graph
            # with tf.device('/job:localhost/replica:0/task:0/device:GPU:2'):
            with tf.variable_scope(self.main_scope):
                self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v, self.std = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)
            
            # Target value network
            with tf.variable_scope(self.target_scope):
                _, _, _, _, _, _, _, self.v_targ, _  = actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)

            # Experience buffer
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

            # Min Double-Q:
            min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)

            # Targets for Q and V regression
            q_backup = tf.stop_gradient(self.r_ph + gamma*(1-self.d_ph)*self.v_targ)
            v_backup = tf.stop_gradient(min_q_pi - alpha * self.logp_pi)

            # Soft actor-critic losses
            pi_loss = tf.reduce_mean(alpha * self.logp_pi - self.q1_pi)
            q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
            v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
            value_loss = q1_loss + q2_loss + v_loss

            # Policy train op 
            # (has to be separate from value train op, because q1_pi appears in pi_loss)
            pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars(self.var_scope('pi')))

            # Value train op
            # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
            value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            value_params = get_vars(self.var_scope('q')) + get_vars(self.var_scope('v'))
            with tf.control_dependencies([train_pi_op]):
                train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

            # Polyak averaging for target variables
            # (control flow because sess.run otherwise evaluates in nondeterministic order)
            with tf.control_dependencies([train_value_op]):
                target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                          for v_main, v_targ in zip(get_vars(self.main_scope), get_vars(self.target_scope))])

            # All ops to call during one training step
            self.step_ops = [pi_loss, q1_loss, q2_loss, v_loss, self.q1, self.q2, self.v, self.logp_pi, 
                        train_pi_op, train_value_op, target_update]

            # Initializing targets to match main variables
            target_init = tf.group([tf.assign(self.v_targ, v_main)
                                      for v_main, self.v_targ in zip(get_vars(self.main_scope), get_vars(self.target_scope))])

            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(target_init)


    def reward(self, env, r, o):
        if len(self.reward_fn) == 0:
            return r
        
        # use self.normalization_factors to normalize the state.
        tup = tuple(reacher_utils.discretize_state(o, self.normalization_factors))
        return self.reward_fn[tup]

    def get_action(self, o, deterministic=False): 
        if self.learn_reduced:
            o = reacher_utils.convert_obs(o)
        with self.graph.as_default():
            act_op = self.mu if deterministic else self.pi
            action = self.sess.run(act_op, feed_dict={self.x_ph: o.reshape(1,-1)})[0]
            return action
        
    def get_sigma(self, o):
        if self.learn_reduced:
            o = reacher_utils.convert_obs(o)
        with self.graph.as_default():
            return self.sess.run(self.std, feed_dict={self.x_ph: o.reshape(1,-1)})[0]

    def test_agent(self, T, n=10, initial_state=[], store_log=True, deterministic=True, reset=False):
        
        denom = 0

        for j in range(n):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            
            if len(initial_state) > 0:
                qpos = initial_state[:len(reacher_utils.qpos)]
                qvel = initial_state[len(reacher_utils.qpos):]
                self.test_env.env.set_state(qpos, qvel)
                o = self.test_env.env._get_obs()
            
            o = reacher_utils.get_state(self.test_env, o)
            while not(d or (ep_len == T)):
                # Take deterministic actions at test time 
                a = self.get_action(o, deterministic)
                o, r, d, _ = self.test_env.step(a)
                o = reacher_utils.get_state(self.test_env, o)

                r = self.reward(self.test_env, r, o)
                ep_ret += r
                ep_len += 1
                denom += 1
                
                if d and reset:
                    d = False

            if store_log:
                self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


    def test_agent_random(self, T, normalization_factors=[], n=10):
        
        p = np.zeros(shape=(tuple(reacher_utils.num_states)))
        p_joint0 = np.zeros(shape=(tuple(reacher_utils.num_states_2d)))
        p_joint1 = np.zeros(shape=(tuple(reacher_utils.num_states_2d)))
        
        cumulative_states_visited_baseline = 0
        states_visited_baseline = []
        cumulative_states_visited_joint0_baseline = 0
        states_visited_joint0_baseline = []
        cumulative_states_visited_joint1_baseline = 0
        states_visited_joint1_baseline = []

        denom = 0

        for j in range(n):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            o = reacher_utils.get_state(self.test_env, o)
            while not(d or (ep_len == T)):
                a = self.test_env.action_space.sample()
                o, r, d, _ = self.test_env.step(a)
                o = reacher_utils.get_state(self.test_env, o)
                r = self.reward(self.test_env, r, o)
                
                # if this is the first time you are seeing this state, increment.
                if p[tuple(reacher_utils.discretize_state(o, normalization_factors))] == 0:
                    cumulative_states_visited_baseline += 1
                states_visited_baseline.append(cumulative_states_visited_baseline)
                if p_joint0[tuple(reacher_utils.discretize_state_2d(o, reacher_utils.joint0th, reacher_utils.joint0v, normalization_factors))]  == 0:
                    cumulative_states_visited_joint0_baseline += 1
                states_visited_joint0_baseline.append(cumulative_states_visited_joint0_baseline)
                
                if p_joint1[tuple(reacher_utils.discretize_state_2d(o, reacher_utils.joint1th, reacher_utils.joint1v, normalization_factors))]  == 0:
                    cumulative_states_visited_joint1_baseline += 1
                states_visited_joint1_baseline.append(cumulative_states_visited_joint1_baseline)
                
                p[tuple(reacher_utils.discretize_state(o, normalization_factors))] += 1
                p_joint0[tuple(reacher_utils.discretize_state_2d(o, reacher_utils.joint0th, reacher_utils.joint0v, normalization_factors))] += 1
                p_joint1[tuple(reacher_utils.discretize_state_2d(o, reacher_utils.joint1th, reacher_utils.joint1v, normalization_factors))] += 1
                
                denom += 1
                ep_len += 1
                
                # CRITICAL: ignore done signal
                if d:
                    d = False
        
        p /= float(denom)
        p_joint0 /= float(denom)
        p_joint1 /= float(denom)
        
        return p, p_joint0, p_joint1, states_visited_baseline, states_visited_joint0_baseline, states_visited_joint1_baseline

    def soft_actor_critic(self, initial_state=[], steps_per_epoch=5000, epochs=100,
            batch_size=100, start_steps=10000, save_freq=1):
        
        with self.graph.as_default():

            # Count variables
            var_counts = tuple(core.count_vars(scope) for scope in 
                               [self.var_scope('pi'), self.var_scope('q1'), self.var_scope('q2'), self.var_scope('v'), self.main_scope])
            print(('\nNumber of parameters: \t pi: %d, \t' + \
                   'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

            # Setup model saving
            self.logger.setup_tf_saver(self.sess, inputs={'x': self.x_ph, 'a': self.a_ph}, 
                                        outputs={'mu': self.mu, 'pi': self.pi, 'q1': self.q1, 'q2': self.q2, 'v': self.v})

            o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
            if len(initial_state) > 0:
                qpos = initial_state[:len(reacher_utils.qpos)]
                qvel = initial_state[len(reacher_utils.qpos):]
                self.env.env.set_state(qpos, qvel)
                o = self.env.env._get_obs()
            
            o = reacher_utils.get_state(self.env, o)

            total_steps = steps_per_epoch * epochs

            # Main loop: collect experience in env and update/log each epoch
            for t in range(total_steps):

                """
                Until start_steps have elapsed, randomly sample actions
                from a uniform distribution for better exploration. Afterwards, 
                use the learned policy. 
                """
                if t > start_steps:
                    if t == start_steps + 1:
                        print("!!!! using policy !!!!")
                    a = self.get_action(o)
                else:
                    a = self.env.action_space.sample()

                # Step the env
                o2, r, d, _ = self.env.step(a)
                o2 = reacher_utils.get_state(self.env, o2)
                r = self.reward(self.env, r, o2)

                # TODO: cap velocity.
                # TODO: something with the way I am converting range of theta to circular?
                
                ep_ret += r
                ep_len += 1

                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
                d = False if ep_len == self.max_ep_len else d

                # Store experience to replay buffer
                if self.learn_reduced:
                    self.replay_buffer.store(reacher_utils.convert_obs(o), 
                                             a, r, reacher_utils.convert_obs(o2), d)
                else:
                    self.replay_buffer.store(o, a, r, o2, d)

                # Super critical: update most recent observation.
                o = o2

                if d or (ep_len == self.max_ep_len):

                    """
                    Perform all SAC updates at the end of the trajectory.
                    This is a slight difference from the SAC specified in the
                    original paper.
                    """
                    for j in range(ep_len):
                        batch = self.replay_buffer.sample_batch(batch_size)
                        feed_dict = {self.x_ph: batch['obs1'],
                                     self.x2_ph: batch['obs2'],
                                     self.a_ph: batch['acts'],
                                     self.r_ph: batch['rews'],
                                     self.d_ph: batch['done'],
                                    }
                        outs = self.sess.run(self.step_ops, feed_dict)
                        self.logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                                     LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                                     VVals=outs[6], LogPi=outs[7])

                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
                    if len(initial_state) > 0:
                        qpos = initial_state[:len(reacher_utils.qpos)]
                        qvel = initial_state[len(reacher_utils.qpos):]
                        self.env.env.set_state(qpos, qvel)
                        o = self.env.env._get_obs()
                    o = reacher_utils.get_state(self.env, o)

                # End of epoch wrap-up
                if t > 0 and t % steps_per_epoch == 0:
                    epoch = t // steps_per_epoch

                    # Save model
                    if (epoch % save_freq == 0) or (epoch == epochs-1):
                        self.logger.save_state({'env': self.env}, None)

                    # Test the performance of the deterministic version of the agent.
                    self.test_agent(self.max_ep_len)

                    # Log info about epoch
                    self.logger.log_tabular('Epoch', epoch)
                    self.logger.log_tabular('EpRet', with_min_and_max=False)
                    self.logger.log_tabular('TestEpRet', with_min_and_max=False)
                    self.logger.log_tabular('EpLen', average_only=True)
                    self.logger.log_tabular('TestEpLen', average_only=True)
                    self.logger.log_tabular('LossPi', average_only=True)
                    self.logger.log_tabular('LossQ1', average_only=True)
                    self.logger.log_tabular('LossQ2', average_only=True)
                    self.logger.log_tabular('LossV', average_only=True)
                    self.logger.dump_tabular()


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=300)
#     parser.add_argument('--l', type=int, default=1)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='sac')
#     args = parser.parse_args()

#     from spinup.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     sac1 = ReacherSoftActorCritic(lambda : gym.make(args.env), 
#         actor_critic=core.mlp_actor_critic, 
#         seed=args.seed, gamma=args.gamma, 
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
#         logger_kwargs=logger_kwargs)
#     sac1.soft_actor_critic(epochs=args.epochs)

#     print("---------- SAC 2 ---------")

#     sac2 = ReacherSoftActorCritic(lambda : gym.make(args.env), 
#         actor_critic=core.mlp_actor_critic, 
#         seed=args.seed, gamma=args.gamma, 
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
#         logger_kwargs=logger_kwargs)
#     sac2.soft_actor_critic(epochs=args.epochs)
