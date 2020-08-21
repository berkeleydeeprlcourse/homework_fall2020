import numpy as np
import tensorflow as tf
import time

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import *
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *

class BCAgent(BaseAgent):
    def __init__(self, sess, env, agent_params):
        super(BCAgent, self).__init__()

        # init vars
        self.env = env
        self.sess = sess
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicySL(sess,
                               self.agent_params['ac_dim'],
                               self.agent_params['ob_dim'],
                               self.agent_params['n_layers'],
                               self.agent_params['size'],
                               discrete = self.agent_params['discrete'],
                               learning_rate = self.agent_params['learning_rate'],
                               ) ## TODO: look in here and implement this

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        self.actor.update(ob_no, ac_na) ## TODO: look in here and implement this

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size) ## TODO: look in here and implement this