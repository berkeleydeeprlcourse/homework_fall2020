import numpy as np
import tensorflow as tf
from .base_policy import BasePolicy
import tensorflow_probability as tfp
import pickle

class Loaded_Gaussian_Policy(BasePolicy):
    def __init__(self, sess, filename, **kwargs):
        super().__init__(**kwargs)

        self.sess = sess

        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())

        self.nonlin_type = data['nonlin_type']
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
        self.policy_params = data[policy_type]

        assert set(self.policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

        self.build_graph()

    ##################################

    def build_graph(self):
        self.define_placeholders()
        self.define_forward_pass()

    ##################################

    def define_placeholders(self):
        self.obs_bo = tf.placeholder(tf.float32, [None, None])

    def define_forward_pass(self):

        # Build the policy. First, observation normalization.
        assert list(self.policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = self.policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
        normedobs_bo = (self.obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6)
        curr_activations_bd = normedobs_bo

        # Hidden layers next
        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = self.read_layer(l)
            curr_activations_bd = self.apply_nonlin(tf.matmul(curr_activations_bd, W) + b)

        # Output layer
        W, b = self.read_layer(self.policy_params['out'])
        self.output_bo = tf.matmul(curr_activations_bd, W) + b

    def read_layer(self, l):
        assert list(l.keys()) == ['AffineLayer']
        assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
        return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

    def apply_nonlin(self, x):
        if self.nonlin_type == 'lrelu':
            return lrelu(x, leak=.01)
        elif self.nonlin_type == 'tanh':
            return tf.tanh(x)
        else:
            raise NotImplementedError(self.nonlin_type)

    ##################################

    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        print("\n\nThis policy class simply loads in a particular type of policy and queries it.")
        print("Not training procedure has been written, so do not try to train it.\n\n")
        raise NotImplementedError

    def get_action(self, obs):
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None, :]
        return self.sess.run(self.output_bo, feed_dict={self.obs_bo : observation})

