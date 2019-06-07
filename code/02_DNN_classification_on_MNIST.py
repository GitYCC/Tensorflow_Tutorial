#!/usr/local/bin/python3.6

import random

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class DNNLogisticClassification:

    def __init__(self, n_features, n_labels,
                 learning_rate=0.5, n_hidden=1000, activation=tf.nn.relu,
                 dropout_ratio=0.5, alpha=0.0):

        self.n_features = n_features
        self.n_labels = n_labels

        self.weights = None
        self.biases = None

        self.graph = tf.Graph()  # initialize new graph
        self.build(learning_rate, n_hidden, activation,
                   dropout_ratio, alpha)  # building graph
        self.sess = tf.Session(graph=self.graph)  # create session by the graph

    def build(self, learning_rate, n_hidden, activation, dropout_ratio, alpha):
        # Building Graph
        with self.graph.as_default():
            ### Input
            self.train_features = tf.placeholder(tf.float32, shape=(None, self.n_features))
            self.train_labels = tf.placeholder(tf.int32, shape=(None, self.n_labels))

            ### Optimalization
            # build neurel network structure and get their predictions and loss
            self.y_, self.original_loss = self.structure(features=self.train_features,
                                                         labels=self.train_labels,
                                                         n_hidden=n_hidden,
                                                         activation=activation,
                                                         dropout_ratio=dropout_ratio,
                                                         train=True)
            # regularization loss
            self.regularization = \
                tf.reduce_sum([tf.nn.l2_loss(w) for w in self.weights.values()]) \
                / tf.reduce_sum([tf.size(w, out_type=tf.float32) for w in self.weights.values()])

            # total loss
            self.loss = self.original_loss + alpha * self.regularization

            # define training operation
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

            ### Prediction
            self.new_features = tf.placeholder(tf.float32, shape=(None, self.n_features))
            self.new_labels = tf.placeholder(tf.int32, shape=(None, self.n_labels))
            self.new_y_, self.new_original_loss = self.structure(features=self.new_features,
                                                                 labels=self.new_labels,
                                                                 n_hidden=n_hidden,
                                                                 activation=activation)
            self.new_loss = self.new_original_loss + alpha * self.regularization

            ### Initialization
            self.init_op = tf.global_variables_initializer()

    def structure(self, features, labels, n_hidden, activation, dropout_ratio=0, train=False):
        # build neurel network structure and return their predictions and loss
        ### Variable
        if (not self.weights) or (not self.biases):
            self.weights = {
                'fc1': tf.Variable(tf.truncated_normal(shape=(self.n_features, n_hidden))),
                'fc2': tf.Variable(tf.truncated_normal(shape=(n_hidden, self.n_labels))),
            }
            self.biases = {
                'fc1': tf.Variable(tf.zeros(shape=(n_hidden))),
                'fc2': tf.Variable(tf.zeros(shape=(self.n_labels))),
            }
        ### Structure
        # layer 1
        fc1 = self.getDenseLayer(features, self.weights['fc1'],
                                 self.biases['fc1'], activation=activation)
        if train:
            fc1 = tf.nn.dropout(fc1, keep_prob=1-dropout_ratio)

        # layer 2
        logits = self.getDenseLayer(fc1, self.weights['fc2'], self.biases['fc2'])

        y_ = tf.nn.softmax(logits)

        loss = tf.reduce_mean(
                 tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        return (y_, loss)

    def getDenseLayer(self, input_layer, weight, bias, activation=None):
        # fully connected layer
        x = tf.add(tf.matmul(input_layer, weight), bias)
        if activation:
            x = activation(x)
        return x

    def fit(self, X, y, epochs=10, validation_data=None, test_data=None, batch_size=None):
        X = self._check_array(X)
        y = self._check_array(y)

        N = X.shape[0]
        random.seed(9000)
        if not batch_size:
            batch_size = N

        self.sess.run(self.init_op)
        for epoch in range(epochs):
            print('Epoch %2d/%2d: ' % (epoch+1, epochs))

            # mini-batch gradient descent
            index = [i for i in range(N)]
            random.shuffle(index)
            while len(index) > 0:
                index_size = len(index)
                batch_index = [index.pop() for _ in range(min(batch_size, index_size))]

                feed_dict = {
                    self.train_features: X[batch_index, :],
                    self.train_labels: y[batch_index],
                }
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                print('[%d/%d] loss = %9.4f     ' % (N-len(index), N, loss), end='\r')

            # evaluate at the end of this epoch
            y_ = self.predict(X)
            train_loss = self.evaluate(X, y)
            train_acc = self.accuracy(y_, y)
            msg = '[%d/%d] loss = %8.4f, acc = %3.2f%%' % (N, N, train_loss, train_acc*100)

            if validation_data:
                val_loss = self.evaluate(validation_data[0], validation_data[1])
                val_acc = self.accuracy(self.predict(validation_data[0]), validation_data[1])
                msg += ', val_loss = %8.4f, val_acc = %3.2f%%' % (val_loss, val_acc*100)

            print(msg)

        if test_data:
            test_acc = self.accuracy(self.predict(test_data[0]), test_data[1])
            print('test_acc = %3.2f%%' % (test_acc*100))

    def accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])

    def predict(self, X):
        X = self._check_array(X)
        return self.sess.run(self.new_y_, feed_dict={self.new_features: X})

    def evaluate(self, X, y):
        X = self._check_array(X)
        y = self._check_array(y)
        return self.sess.run(self.new_loss, feed_dict={self.new_features: X,
                                                       self.new_labels: y})

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

    model = DNNLogisticClassification(
        n_features=28*28,
        n_labels=10,
        learning_rate=0.5,
        n_hidden=1000,
        activation=tf.nn.relu,
        dropout_ratio=0.5,
        alpha=0.01,
    )
    model.fit(
        X=train_data.images,
        y=train_data.labels,
        epochs=3,
        validation_data=(valid_data.images, valid_data.labels),
        test_data=(test_data.images, test_data.labels),
        batch_size=32,
    )
