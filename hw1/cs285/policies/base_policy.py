import numpy as np

class BasePolicy(object):

    def __init__(self, **kwargs):
       super(BasePolicy, self).__init__(**kwargs)

    def build_graph(self):
        raise NotImplementedError

    def get_action(self, obs):
        raise NotImplementedError

    def update(self, obs, acs):
        raise NotImplementedError

    def save(self, filepath):
    	raise NotImplementedError

    def restore(self, filepath):
    	raise NotImplementedError