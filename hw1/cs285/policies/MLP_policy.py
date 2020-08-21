import numpy as np
import tensorflow as tf
from .base_policy import BasePolicy
from cs285.infrastructure.tf_utils import build_mlp
import tensorflow_probability as tfp

class MLPPolicy(BasePolicy):

    def __init__(self,
        sess,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        policy_scope='policy_vars',
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training

        # build TF graph
        with tf.variable_scope(policy_scope, reuse=tf.AUTO_REUSE):
            self.build_graph()

        # saver for policy variables that are not related to training
        self.policy_vars = [v for v in tf.all_variables() if policy_scope in v.name and 'train' not in v.name]
        self.policy_saver = tf.train.Saver(self.policy_vars, max_to_keep=None)

    ##################################

    def build_graph(self):
        self.define_placeholders()
        self.define_forward_pass()
        self.build_action_sampling()
        if self.training:
            with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
                self.define_train_op()

    ##################################

    def define_placeholders(self):
        raise NotImplementedError

    def define_forward_pass(self):
        # TODO implement this build_mlp function in tf_utils
        mean = build_mlp(self.observations_pl, output_size=self.ac_dim, scope='continuous_logits', n_layers=self.n_layers, size=self.size)
        logstd = tf.Variable(tf.zeros(self.ac_dim), name='logstd')
        self.parameters = (mean, logstd)

    def build_action_sampling(self):
        mean, logstd = self.parameters
        self.sample_ac = mean + tf.exp(logstd) * tf.random_normal(tf.shape(mean), 0, 1)

    def define_train_op(self):
        raise NotImplementedError

    ##################################

    def save(self, filepath):
        self.policy_saver.save(self.sess, filepath, write_meta_graph=False)

    def restore(self, filepath):
        self.policy_saver.restore(self.sess, filepath)

    ##################################

    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):

        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # HINT1: you will need to call self.sess.run
        # HINT2: the tensor we're interested in evaluating is self.sample_ac
        # HINT3: in order to run self.sample_ac, it will need observation fed into the feed_dict
        return TODO

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    def define_placeholders(self):
        # placeholder for observations
        self.observations_pl = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

        # placeholder for actions
        self.actions_pl = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

        if self.training:
            self.acs_labels_na = tf.placeholder(shape=[None, self.ac_dim], name="labels", dtype=tf.float32)

    def define_train_op(self):
        true_actions = self.acs_labels_na
        predicted_actions = self.sample_ac

        # TODO define the loss that will be used to train this policy
        # HINT1: remember that we are doing supervised learning
        # HINT2: use tf.losses.mean_squared_error
        self.loss = TODO
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def update(self, observations, actions):
        assert(self.training, 'Policy must be created with training=True in order to perform training updates...')
        self.sess.run(self.train_op, feed_dict={self.observations_pl: observations, self.acs_labels_na: actions})

