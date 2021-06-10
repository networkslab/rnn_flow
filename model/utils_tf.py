from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.sparse as sp


class AGCGRU:

    def __init__(self, adj_mx, input_dim, embed_dim, num_units, max_diffusion_step, scope):

        self._num_nodes = adj_mx.shape[0]
        self._num_units = num_units
        self._embed_dim = embed_dim
        self._max_diffusion_step = max_diffusion_step

        self.num_matrices = self._max_diffusion_step + 1
        self.input_size = input_dim + num_units

        with tf.variable_scope(scope):

            self.Wr = tf.get_variable('Wr', [self._embed_dim, self.num_matrices, self.input_size, self._num_units],
                                      dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            self.Wu = tf.get_variable('Wu', [self._embed_dim, self.num_matrices, self.input_size, self._num_units],
                                      dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            self.Wc = tf.get_variable('Wc', [self._embed_dim, self.num_matrices, self.input_size, self._num_units],
                                      dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            self.br = tf.get_variable("br", [self._embed_dim, self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(1.0, dtype=tf.float32))

            self.bu = tf.get_variable("bu", [self._embed_dim, self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(1.0, dtype=tf.float32))

            self.bc = tf.get_variable("bc", [self._embed_dim, self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32))

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _replicate_input(self, x, nParticle):
        if len(x.get_shape()) < 4:
            return tf.tile(x[:, :, :, tf.newaxis], [1, 1, 1, nParticle])
        else:
            return x

    def forward(self, inputs, state, node_embedding):
        graph_processed_input = self._graph_filtering(inputs, state, node_embedding)

        r = tf.nn.sigmoid(self._gconv(graph_processed_input, node_embedding, self.Wr, self.br))
        u = tf.nn.sigmoid(self._gconv(graph_processed_input, node_embedding, self.Wu, self.bu))

        graph_processed_input_times_r = self._graph_filtering(inputs, r * state, node_embedding)

        c = tf.nn.tanh(self._gconv(graph_processed_input_times_r, node_embedding, self.Wc, self.bc))
        output = u * state + (1 - u) * c
        return output

    def _gconv(self, graph_processed_input, node_embedding,  weights, biases):

        batch_size = graph_processed_input.get_shape()[0].value
        nParticle = graph_processed_input.get_shape()[1].value

        weights_ = tf.einsum('nd, dkio -> nkio', node_embedding, weights)
        biases_ = tf.matmul(node_embedding, biases)

        x = tf.einsum('bpnik, nkio -> bpno', graph_processed_input, weights_)  # (batch_size, nParticle, self._num_nodes, output_size)
        biases_ = tf.tile(biases_[tf.newaxis, tf.newaxis, :, :], [batch_size, nParticle, 1, 1])
        x += biases_
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node, output_size, nParticle)
        return tf.transpose(x, perm=[0, 2, 3, 1])

    def _graph_filtering(self, inputs, state, node_embedding):

        adj_learned = tf.nn.softmax(tf.nn.relu(tf.matmul(node_embedding, tf.transpose(node_embedding, perm=[1, 0]))))
        supports = [adj_learned]

        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim, nParticle)
        batch_size = inputs.get_shape()[0].value
        nParticle = state.get_shape()[3].value
        inputs = self._replicate_input(inputs, nParticle)

        inputs = tf.reshape(inputs, shape=[batch_size, self._num_nodes, -1, nParticle])
        state = tf.reshape(state, shape=[batch_size, self._num_nodes, -1, nParticle])
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0, 3])  # (num_nodes, total_arg_size, batch_size, nParticle)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size * nParticle])
        x = tf.expand_dims(x0, axis=0)

        for support in supports:
            x1 = tf.matmul(support, x0)
            x = self._concat(x, x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * tf.matmul(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = tf.reshape(x, shape=[self.num_matrices, self._num_nodes, input_size, batch_size, nParticle])
        x = tf.transpose(x, perm=[3, 4, 1, 2, 0])  # (batch_size, nParticle, num_nodes, input_size, order)
        return x


class DCGRU:

    def __init__(self, adj_mx, input_dim, num_units, max_diffusion_step, scope):

        self._num_nodes = adj_mx.shape[0]
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []

        supports = []

        supports.append(calculate_random_walk_matrix(adj_mx).T)
        supports.append(calculate_random_walk_matrix(adj_mx.T).T)

        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self.num_matrices = len(self._supports) * self._max_diffusion_step + 1
        self.input_size = input_dim + num_units

        with tf.variable_scope(scope):

            self.Wr = tf.get_variable('Wr', [self.input_size * self.num_matrices, self._num_units], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

            self.Wu = tf.get_variable('Wu', [self.input_size * self.num_matrices, self._num_units], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

            self.Wc = tf.get_variable('Wc', [self.input_size * self.num_matrices, self._num_units], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

            self.br = tf.get_variable("br", [self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(1.0, dtype=tf.float32))

            self.bu = tf.get_variable("bu", [self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(1.0, dtype=tf.float32))
            self.bc = tf.get_variable("bc", [self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _replicate_input(self, x, nParticle):
        if len(x.get_shape()) < 4:
            return tf.tile(x[:, :, :, tf.newaxis], [1, 1, 1, nParticle])
        else:
            return x

    def forward(self, inputs, state):
        r = tf.nn.sigmoid(self._gconv(inputs, state, self._num_units, self.Wr, self.br))
        u = tf.nn.sigmoid(self._gconv(inputs, state, self._num_units, self.Wu, self.bu))
        c = tf.nn.tanh(self._gconv(inputs, r * state, self._num_units, self.Wc, self.bc))
        output = u * state + (1 - u) * c
        return output

    def _gconv(self, inputs, state, output_size, weights, biases):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim, nParticle)
        batch_size = inputs.get_shape()[0].value
        nParticle = state.get_shape()[3].value
        inputs = self._replicate_input(inputs, nParticle)

        inputs = tf.reshape(inputs, shape=[batch_size, self._num_nodes, -1, nParticle])
        state = tf.reshape(state, shape=[batch_size, self._num_nodes, -1, nParticle])
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0, 3])  # (num_nodes, total_arg_size, batch_size, nParticle)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size * nParticle])
        x = tf.expand_dims(x0, axis=0)

        for support in self._supports:
            x1 = tf.sparse_tensor_dense_matmul(support, x0)
            x = self._concat(x, x1)

            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = tf.reshape(x, shape=[self.num_matrices, self._num_nodes, input_size, batch_size, nParticle])
        x = tf.transpose(x, perm=[3, 4, 1, 2, 0])  # (batch_size, nParticle, num_nodes, input_size, order)
        x = tf.reshape(x, shape=[batch_size * nParticle * self._num_nodes, input_size * self.num_matrices])

        x = tf.matmul(x, weights)  # (batch_size * nParticle * self._num_nodes, output_size)

        x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node, output_size, nParticle)
        return tf.transpose(tf.reshape(x, [batch_size, nParticle, self._num_nodes, output_size]), perm=[0, 2, 3, 1])


class GRU:

    def __init__(self, adj_mx, input_dim, num_units, scope):

        self._num_nodes = adj_mx.shape[0]
        self._num_units = num_units
        self.input_size = input_dim + num_units

        with tf.variable_scope(scope):

            self.Wr = tf.get_variable('Wr', [self.input_size, self._num_units], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

            self.Wu = tf.get_variable('Wu', [self.input_size, self._num_units], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

            self.Wc = tf.get_variable('Wc', [self.input_size, self._num_units], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

            self.br = tf.get_variable("br", [self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(1.0, dtype=tf.float32))

            self.bu = tf.get_variable("bu", [self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(1.0, dtype=tf.float32))
            self.bc = tf.get_variable("bc", [self._num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32))

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _replicate_input(self, x, nParticle):
        if len(x.get_shape()) < 4:
            return tf.tile(x[:, :, :, tf.newaxis], [1, 1, 1, nParticle])
        else:
            return x

    def forward(self, inputs, state):
        r = tf.nn.sigmoid(self._conv(inputs, state, self._num_units, self.Wr, self.br))
        u = tf.nn.sigmoid(self._conv(inputs, state, self._num_units, self.Wu, self.bu))
        c = tf.nn.tanh(self._conv(inputs, r * state, self._num_units, self.Wc, self.bc))
        output = u * state + (1 - u) * c
        return output

    def _conv(self, inputs, state, output_size, weights, biases):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim, nParticle)
        batch_size = inputs.get_shape()[0].value
        nParticle = state.get_shape()[3].value
        inputs = self._replicate_input(inputs, nParticle)

        inputs = tf.reshape(inputs, shape=[batch_size, self._num_nodes, -1, nParticle])
        state = tf.reshape(state, shape=[batch_size, self._num_nodes, -1, nParticle])
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value

        x = inputs_and_state
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # (batch_size, nParticle, num_nodes, input_size)
        x = tf.reshape(x, shape=[batch_size * nParticle * self._num_nodes, input_size])

        x = tf.matmul(x, weights)  # (batch_size * nParticle * self._num_nodes, output_size)

        x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node, output_size, nParticle)
        return tf.transpose(tf.reshape(x, [batch_size, nParticle, self._num_nodes, output_size]), perm=[0, 2, 3, 1])


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def linear_projection(state, weight, weight_delta, noise=True):

    num_nodes = state.get_shape()[1].value
    num_units = state.get_shape()[2].value
    nParticle = state.get_shape()[3].value
    nBatch = state.get_shape()[0].value

    state = tf.reshape(tf.transpose(state, perm=(0, 1, 3, 2)), shape=(-1, num_units))
    measurement_mean = tf.reshape(tf.matmul(state, weight), shape=(nBatch, num_nodes, nParticle))
    noise_std_dev = tf.reshape(tf.math.softplus(tf.matmul(state, weight_delta)), shape=(nBatch, num_nodes, nParticle))

    measurement_mean = tf.where(tf.is_nan(measurement_mean), tf.zeros_like(measurement_mean), measurement_mean)
    measurement_mean = tf.where(tf.is_inf(measurement_mean), tf.zeros_like(measurement_mean), measurement_mean)

    noise_std_dev = tf.where(tf.is_nan(noise_std_dev), tf.zeros_like(noise_std_dev), noise_std_dev)
    noise_std_dev = tf.where(tf.is_inf(noise_std_dev), tf.zeros_like(noise_std_dev), noise_std_dev)

    if noise:
        measurement = measurement_mean + noise_std_dev * tf.random.normal(shape=(tf.shape(measurement_mean)), mean=0, stddev=1)

    return measurement, measurement_mean, noise_std_dev


def particle_flow(particles, measurement, weight, weight_delta):
    num_nodes = tf.shape(particles)[1]
    num_units = tf.shape(particles)[2]
    nParticle = tf.shape(particles)[3]
    nBatch = tf.shape(particles)[0]

    particles_temp = particles

    lambda_seq = generate_exponential_lambda(29, 1.2)

    mu_0_all = tf.reduce_mean(particles, axis=3)
    P_all = tf_cov(tf.reshape(tf.transpose(particles, perm=(0, 1, 3, 2)), shape=[nBatch, -1, num_units]))

    delta = tf.squeeze(tf.log(1 + tf.exp(tf.matmul(tf.reduce_mean(mu_0_all, axis=1), weight_delta))))
    R = delta * delta
    R_inv = 1.0 / R

    PH_t = tf.einsum('ijk,kl->ijl', P_all, weight)
    PH_tH = tf.einsum('ijk,kl->ijl', PH_t, tf.transpose(weight))
    HPH_t = tf.squeeze(tf.einsum('ij,kjl->ikl', tf.transpose(weight), PH_t))

    PH_t_R_inv_z = tf.tile(R_inv[:, tf.newaxis, tf.newaxis], [1, num_units, num_nodes]) * tf.einsum('ijk,ikl->ijl', PH_t, measurement)

    C = tf.tile(tf.eye(num_units)[tf.newaxis, :, :], [nBatch, 1, 1])
    D = tf.zeros([nBatch, num_units, num_nodes])

    # particle flow
    lmbd_prev = 0

    for lmbd in lambda_seq:
        step_size = lmbd - lmbd_prev

        A_denom = lmbd*HPH_t + R
        A = -0.5 * PH_tH / tf.tile(A_denom[:, tf.newaxis, tf.newaxis], [1, num_units, num_units])

        b_term1 = tf.tile(tf.eye(num_units)[tf.newaxis, :, :], [nBatch, 1, 1]) + lmbd * A
        b_term2 = tf.tile(tf.eye(num_units)[tf.newaxis, :, :], [nBatch, 1, 1]) + 2.0 * lmbd * A
        b_term3 = tf.einsum('ijk,ikl->ijl', b_term1, PH_t_R_inv_z) + tf.einsum('ijk,ilk->ijl', A, mu_0_all)
        b = tf.einsum('ijk,ikl->ijl', b_term2, b_term3)

        C_term1 = tf.tile(tf.eye(num_units)[tf.newaxis, :, :], [nBatch, 1, 1]) + step_size * A
        C = tf.einsum('ijk,ikl->ijl', C_term1, C)
        D = tf.einsum('ijk,ikl->ijl', C_term1, D) + step_size * b

        lmbd_prev = lmbd

    particles = tf.einsum('ijk,ilkp->iljp', C, particles) + \
                tf.transpose(tf.tile(D[:, :, :, tf.newaxis], [1, 1, 1, nParticle]), perm=(0, 2, 1, 3))

    particles = tf.where(tf.is_nan(particles), particles_temp, particles)
    particles = tf.where(tf.is_inf(particles), particles_temp, particles)
    return particles


def tf_cov(x):
    # x: nBatch x nSamples x dim, cov_xx: n_batch x dim x dim
    mean_x = tf.reduce_mean(x, axis=1, keep_dims=True)
    mx = tf.einsum('ijk,ikl->ijl', tf.transpose(mean_x, perm=(0, 2, 1)), mean_x)
    vx = tf.einsum('ijk,ikl->ijl', tf.transpose(x, perm=(0, 2, 1)), x)/tf.cast(tf.shape(x)[1], tf.float32)
    cov_xx = vx - mx
    return cov_xx


def generate_exponential_lambda(num_steps, delta_lambda_ratio):
    lambda_1 = (1-delta_lambda_ratio) / (1-delta_lambda_ratio**num_steps)
    lambda_seq = np.cumsum(lambda_1 * (delta_lambda_ratio**np.arange(num_steps))).astype('float32')
    return lambda_seq


def prediction_batch(particles_0, particles, init_measurement, target, rnn_0, rnn_1, weight, weight_delta, node_embedding,
                        is_training, curriculam_prob, rnn_type):
    nParticle = particles.get_shape()[3].value
    pred_len = target.get_shape()[1].value

    input_dim = target.get_shape()[3].value

    for t in range(pred_len):
        if t == 0:
            if rnn_type == 'agcgru':
                particles_0 = rnn_0.forward(init_measurement, particles_0, node_embedding)
            else:
                particles_0 = rnn_0.forward(init_measurement, particles_0)
        else:
            c = tf.random_uniform((), minval=0, maxval=1.)
            curriculam = tf.cond(tf.less(c, curriculam_prob), lambda: 'yes', lambda: 'no')

            if is_training and curriculam == 'yes':
                if rnn_type == 'agcgru':
                    particles_0 = rnn_0.forward(target[:, t-1, :, :input_dim], particles_0, node_embedding)
                else:
                    particles_0 = rnn_0.forward(target[:, t - 1, :, :input_dim], particles_0)
            else:
                if input_dim == 1:
                    obs_samples_aug = obs_samples_temp[:, :, tf.newaxis, :]
                else:
                    obs_samples_aug = tf.concat([obs_samples_temp[:, :, tf.newaxis, :],
                                         tf.tile(target[:, t-1, :, input_dim-1:][:, :, :, tf.newaxis], [1, 1, 1, nParticle])], axis=2)
                if rnn_type == 'agcgru':
                    particles_0 = rnn_0.forward(obs_samples_aug, particles_0, node_embedding)
                else:
                    particles_0 = rnn_0.forward(obs_samples_aug, particles_0)
        if rnn_type == 'agcgru':
            particles = rnn_1.forward(particles_0, particles, node_embedding)
        else:
            particles = rnn_1.forward(particles_0, particles)
        obs_samples_temp, obs_mu_temp, obs_sigma_temp = linear_projection(particles, weight, weight_delta, noise=True)

        if t == 0:
            prediction_samples = obs_samples_temp[:, tf.newaxis, :, :]
            prediction_mu = obs_mu_temp[:, tf.newaxis, :, :]
            prediction_sigma = obs_sigma_temp[:, tf.newaxis, :, :]
        else:
            prediction_samples = tf.concat([prediction_samples, obs_samples_temp[:, tf.newaxis, :, :]], axis=1)
            prediction_mu = tf.concat([prediction_mu, obs_mu_temp[:, tf.newaxis, :, :]], axis=1)
            prediction_sigma = tf.concat([prediction_sigma, obs_sigma_temp[:, tf.newaxis, :, :]], axis=1)

    return prediction_samples, prediction_mu, prediction_sigma


def training(inputs, targets, rnn_0, rnn_1, weight, weight_delta,
                 rho, node_embedding, num_units, nParticle, is_training, curriculam_prob, rnn_type):

    batch = inputs.get_shape()[0].value
    seq_len = inputs.get_shape()[1].value
    num_nodes = inputs.get_shape()[2].value
    input_dim = inputs.get_shape()[3].value

    particles_0 = rho * tf.random.normal(shape=[batch, num_nodes, num_units, nParticle])
    particles = rho * tf.random.normal(shape=[batch, num_nodes, num_units, nParticle])

    for t in range(seq_len):
        if t == 0:
            if rnn_type == 'agcgru':
                particles_0 = rnn_0.forward(tf.zeros(shape=[batch, num_nodes, input_dim]), particles_0, node_embedding)
            else:
                particles_0 = rnn_0.forward(tf.zeros(shape=[batch, num_nodes, input_dim]), particles_0)
        else:
            if rnn_type == 'agcgru':
                particles_0 = rnn_0.forward(inputs[:, t-1, :, :input_dim], particles_0, node_embedding)
            else:
                particles_0 = rnn_0.forward(inputs[:, t - 1, :, :input_dim], particles_0)

        if rnn_type == 'agcgru':
            particles = rnn_1.forward(particles_0, particles, node_embedding)
        else:
            particles = rnn_1.forward(particles_0, particles)
        particles = particle_flow(particles, inputs[:, t:t+1, :, 0], weight, weight_delta)

    prediction_samples, prediction_mu, prediction_sigma = prediction_batch(particles_0, particles, inputs[:, -1, :, :], targets, rnn_0, rnn_1, weight,
                                             weight_delta, node_embedding, is_training, curriculam_prob, rnn_type)
    return prediction_samples, prediction_mu, prediction_sigma