#!/usr/local/bin/python3.6

import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class Autoencoder:

    def __init__(self, n_features, learning_rate=0.5, n_hidden=[1000, 500, 250, 2], alpha=0.0):
        self.n_features = n_features

        self.weights = None
        self.biases = None

        self.graph = tf.Graph()  # initialize new grap
        self.build(n_features, learning_rate, n_hidden, alpha)  # building graph
        self.sess = tf.Session(graph=self.graph)  # create session by the graph

    def build(self, n_features, learning_rate, n_hidden, alpha):
        with self.graph.as_default():
            ### Input
            self.train_features = tf.placeholder(tf.float32, shape=(None, n_features))
            self.train_targets = tf.placeholder(tf.float32, shape=(None, n_features))

            ### Optimalization
            # build neurel network structure and get their predictions and loss
            self.y_, self.original_loss, _ = self.structure(
                                               features=self.train_features,
                                               targets=self.train_targets,
                                               n_hidden=n_hidden)

            # regularization loss
            # weight elimination L2 regularizer
            self.regularizer = \
                tf.reduce_sum([tf.reduce_sum(
                        tf.pow(w, 2)/(1+tf.pow(w, 2))) for w in self.weights.values()]) \
                / tf.reduce_sum(
                    [tf.size(w, out_type=tf.float32) for w in self.weights.values()])

            # total loss
            self.loss = self.original_loss + alpha * self.regularizer

            # define training operation
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            ### Prediction
            self.new_features = tf.placeholder(tf.float32, shape=(None, n_features))
            self.new_targets = tf.placeholder(tf.float32, shape=(None, n_features))
            self.new_y_, self.new_original_loss, self.new_encoder = self.structure(
                                                          features=self.new_features,
                                                          targets=self.new_targets,
                                                          n_hidden=n_hidden)
            self.new_loss = self.new_original_loss + alpha * self.regularizer

            ### Initialization
            self.init_op = tf.global_variables_initializer()

    def structure(self, features, targets, n_hidden):
        ### Variable
        if (not self.weights) and (not self.biases):
            self.weights = {}
            self.biases = {}

            n_encoder = [self.n_features]+n_hidden
            for i, n in enumerate(n_encoder[:-1]):
                self.weights['encode{}'.format(i+1)] = \
                    tf.Variable(tf.truncated_normal(
                        shape=(n, n_encoder[i+1]), stddev=0.1), dtype=tf.float32)
                self.biases['encode{}'.format(i+1)] = \
                    tf.Variable(tf.zeros(shape=(n_encoder[i+1])), dtype=tf.float32)

            n_decoder = list(reversed(n_hidden))+[self.n_features]
            for i, n in enumerate(n_decoder[:-1]):
                self.weights['decode{}'.format(i+1)] = \
                    tf.Variable(tf.truncated_normal(
                        shape=(n, n_decoder[i+1]), stddev=0.1), dtype=tf.float32)
                self.biases['decode{}'.format(i+1)] = \
                    tf.Variable(tf.zeros(shape=(n_decoder[i+1])), dtype=tf.float32)

        ### Structure
        activation = tf.nn.relu

        encoder = self.getDenseLayer(features,
                                     self.weights['encode1'],
                                     self.biases['encode1'],
                                     activation=activation)

        for i in range(1, len(n_hidden)-1):
            encoder = self.getDenseLayer(
                encoder,
                self.weights['encode{}'.format(i+1)],
                self.biases['encode{}'.format(i+1)],
                activation=activation,
            )

        encoder = self.getDenseLayer(
            encoder,
            self.weights['encode{}'.format(len(n_hidden))],
            self.biases['encode{}'.format(len(n_hidden))],
        )

        decoder = self.getDenseLayer(encoder,
                                     self.weights['decode1'],
                                     self.biases['decode1'],
                                     activation=activation)

        for i in range(1, len(n_hidden)-1):
            decoder = self.getDenseLayer(
                decoder,
                self.weights['decode{}'.format(i+1)],
                self.biases['decode{}'.format(i+1)],
                activation=activation,
            )

        y_ = self.getDenseLayer(
            decoder,
            self.weights['decode{}'.format(len(n_hidden))],
            self.biases['decode{}'.format(len(n_hidden))],
            activation=tf.nn.sigmoid,
        )

        loss = tf.reduce_mean(tf.pow(targets - y_, 2))

        return (y_, loss, encoder)

    def getDenseLayer(self, input_layer, weight, bias, activation=None):
        x = tf.add(tf.matmul(input_layer, weight), bias)
        if activation:
            x = activation(x)
        return x

    def fit(self, X, Y, epochs=10, validation_data=None, test_data=None, batch_size=None):
        X = self._check_array(X)
        Y = self._check_array(Y)

        N = X.shape[0]
        random.seed(9000)
        if not batch_size:
            batch_size = N

        self.sess.run(self.init_op)
        for epoch in range(epochs):
            print('Epoch %2d/%2d: ' % (epoch+1, epochs))
            start_time = time.time()

            # mini-batch gradient descent
            index = [i for i in range(N)]
            random.shuffle(index)
            while len(index) > 0:
                index_size = len(index)
                batch_index = [index.pop() for _ in range(min(batch_size, index_size))]

                feed_dict = {self.train_features: X[batch_index, :],
                             self.train_targets: Y[batch_index, :]}
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                print('[%d/%d] loss = %9.4f     ' % (N-len(index), N, loss), end='\r')

            # evaluate at the end of this epoch
            msg_valid = ''
            if validation_data is not None:
                val_loss = self.evaluate(validation_data[0], validation_data[1])
                msg_valid = ', val_loss = %9.4f' % (val_loss)

            train_loss = self.evaluate(X, Y)
            print('[%d/%d] %ds loss = %9.4f %s' % (N, N, time.time()-start_time,
                                                   train_loss, msg_valid))

        if test_data is not None:
            test_loss = self.evaluate(test_data[0], test_data[1])
            print('test_loss = %9.4f' % (test_loss))

    def encode(self, X):
        X = self._check_array(X)
        return self.sess.run(self.new_encoder, feed_dict={self.new_features: X})

    def predict(self, X):
        X = self._check_array(X)
        return self.sess.run(self.new_y_, feed_dict={self.new_features: X})

    def evaluate(self, X, Y):
        X = self._check_array(X)
        return self.sess.run(self.new_loss, feed_dict={self.new_features: X,
                                                       self.new_targets: Y})

    def _check_array(self, ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape) == 1:
            ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
        return ndarray


if __name__ == '__main__':
    print('Extract MNIST Dataset ...')

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    train_data = mnist.train
    valid_data = mnist.validation
    test_data = mnist.test

    model_2 = Autoencoder(
        n_features=28*28,
        learning_rate=0.0005,
        n_hidden=[512, 32, 4],
        alpha=0.001,
    )
    model_2.fit(
        X=train_data.images,
        Y=train_data.images,
        epochs=20,
        validation_data=(valid_data.images, valid_data.images),
        test_data=(test_data.images, test_data.images),
        batch_size=8,
    )

    fig, axis = plt.subplots(2, 15, figsize=(15, 2))
    for i in range(0, 15):
        img_original = np.reshape(test_data.images[i], (28, 28))
        axis[0][i].imshow(img_original, cmap='gray')
        img = np.reshape(model_2.predict(test_data.images[i]), (28, 28))
        axis[1][i].imshow(img, cmap='gray')
    plt.show()

    ### get code
    encode = model_2.encode(test_data.images)

    ### PCA 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X = pca.fit_transform(encode)
    Y = np.argmax(test_data.labels, axis=1)

    # plot
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.colorbar()
    plt.show()

    ### TSNE 2D visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(encode)
    Y = np.argmax(test_data.labels, axis=1)

    # plot
    plt.figure(figsize=(10, 8))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y)
    plt.colorbar()
    plt.show()
