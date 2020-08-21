import numpy as np
import tensorflow as tf

class BaseAgent(object):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self):
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError