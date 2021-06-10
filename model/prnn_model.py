from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model.utils_tf import AGCGRU, DCGRU, GRU, training

import tensorflow as tf
import numpy as np


class PRNNModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        nParticle = int(model_kwargs.get('nParticle', 10))
        nParticle_test = int(model_kwargs.get('nParticle_test', 25))

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))
        embed_dim = int(model_kwargs.get('embed_dim', 10))
        rho = float(model_kwargs.get('rho', 1.0))
        rnn_type = model_kwargs.get('rnn_type', 'agcgru')  # agcgru/dcgru/gru

        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')

        node_embedding = tf.get_variable("node_embedding", [num_nodes, embed_dim], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())

        if rnn_type == 'agcgru':
            rnn_0 = AGCGRU(adj_mx=adj_mx, input_dim=input_dim, embed_dim=embed_dim, num_units=rnn_units,
                          max_diffusion_step=max_diffusion_step, scope="layer0")
            rnn_1 = AGCGRU(adj_mx=adj_mx, input_dim=rnn_units, embed_dim=embed_dim, num_units=rnn_units,
                          max_diffusion_step=max_diffusion_step, scope="layer1")
        elif rnn_type == 'dcgru':
            rnn_0 = DCGRU(adj_mx=adj_mx, input_dim=input_dim, num_units=rnn_units,
                          max_diffusion_step=max_diffusion_step, scope="layer0")
            rnn_1 = DCGRU(adj_mx=adj_mx, input_dim=rnn_units, num_units=rnn_units,
                          max_diffusion_step=max_diffusion_step, scope="layer1")
        elif rnn_type == 'gru':
            rnn_0 = GRU(adj_mx=adj_mx, input_dim=input_dim, num_units=rnn_units, scope="layer0")
            rnn_1 = GRU(adj_mx=adj_mx, input_dim=rnn_units, num_units=rnn_units, scope="layer1")
        else:
            print('ERROR! ERROR! ERROR!')
            exit(0)

        weight = tf.get_variable("weight", [rnn_units, output_dim], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        weight_delta = tf.get_variable("weight_delta", [rnn_units, output_dim], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('DCRNN_SEQ'):
            curriculam_prob = self._compute_sampling_threshold(global_step, cl_decay_steps)

            if is_training:
                output_samples, output_mu, output_sigma = training(self._inputs, self._labels, rnn_0, rnn_1, weight, weight_delta,
                                              rho, node_embedding, rnn_units, nParticle, is_training, curriculam_prob, rnn_type)
            else:
                output_samples, output_mu, output_sigma = training(self._inputs, self._labels, rnn_0, rnn_1, weight, weight_delta,
                                              rho, node_embedding, rnn_units, nParticle_test, is_training, curriculam_prob, rnn_type)

        # Project the output to output_dim.
        outputs = tf.reduce_mean(output_samples, axis=3)
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._outputs_samples = output_samples
        self._outputs_mu = output_mu
        self._outputs_sigma = output_sigma
        self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

    @property
    def outputs_samples(self):
        return self._outputs_samples

    @property
    def outputs_mu(self):
        return self._outputs_mu

    @property
    def outputs_sigma(self):
        return self._outputs_sigma
