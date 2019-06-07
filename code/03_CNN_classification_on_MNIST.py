#!/usr/local/bin/python3.6

import random

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class CNNLogisticClassification:

    def __init__(self, shape_picture, n_labels,
                 learning_rate=0.5, dropout_ratio=0.5, alpha=0.0):
        self.shape_picture = shape_picture
        self.n_labels = n_labels

        self.weights = None
        self.biases = None

        self.graph = tf.Graph()  # initialize new grap
        self.build(learning_rate, dropout_ratio, alpha)  # building graph
        self.sess = tf.Session(graph=self.graph)  # create session by the graph

    def build(self, learning_rate, dropout_ratio, alpha):
        with self.graph.as_default():
            ### Input
            self.train_pictures = tf.placeholder(tf.float32,
                                                 shape=[None]+self.shape_picture)
            self.train_labels = tf.placeholder(tf.int32,
                                               shape=(None, self.n_labels))

            ### Optimalization
            # build neurel network structure and get their predictions and loss
            self.y_, self.original_loss = self.structure(pictures=self.train_pictures,
                                                         labels=self.train_labels,
                                                         dropout_ratio=dropout_ratio,
                                                         train=True, )
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
            self.new_pictures = tf.placeholder(tf.float32,
                                               shape=[None]+self.shape_picture)
            self.new_labels = tf.placeholder(tf.int32,
                                             shape=(None, self.n_labels))
            self.new_y_, self.new_original_loss = self.structure(pictures=self.new_pictures,
                                                                 labels=self.new_labels)
            self.new_loss = self.new_original_loss + alpha * self.regularization

            ### Initialization
            self.init_op = tf.global_variables_initializer()

    def structure(self, pictures, labels, dropout_ratio=None, train=False):
        ### Variable
        ## LeNet5 Architecture(http://yann.lecun.com/exdb/lenet/)
        # input:(batch,28,28,1) => conv1[5x5,6] => (batch,24,24,6)
        # pool2 => (batch,12,12,6) => conv2[5x5,16] => (batch,8,8,16)
        # pool4 => fatten5 => (batch,4x4x16) => fc6 => (batch,120)
        # (batch,120) => fc7 => (batch,84)
        # (batch,84) => fc8 => (batch,10) => softmax

        if (not self.weights) and (not self.biases):
            self.weights = {
                'conv1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6),
                                                         stddev=0.1)),
                'conv3': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16),
                                                         stddev=0.1)),
                'fc6': tf.Variable(tf.truncated_normal(shape=(4*4*16, 120),
                                                       stddev=0.1)),
                'fc7': tf.Variable(tf.truncated_normal(shape=(120, 84),
                                                       stddev=0.1)),
                'fc8': tf.Variable(tf.truncated_normal(shape=(84, self.n_labels),
                                                       stddev=0.1)),
            }
            self.biases = {
                'conv1': tf.Variable(tf.zeros(shape=(6))),
                'conv3': tf.Variable(tf.zeros(shape=(16))),
                'fc6': tf.Variable(tf.zeros(shape=(120))),
                'fc7': tf.Variable(tf.zeros(shape=(84))),
                'fc8': tf.Variable(tf.zeros(shape=(self.n_labels))),
            }

        ### Structure
        conv1 = self.getConv2DLayer(pictures,
                                    self.weights['conv1'], self.biases['conv1'],
                                    activation=tf.nn.relu)
        pool2 = tf.nn.max_pool(conv1,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = self.getConv2DLayer(pool2,
                                    self.weights['conv3'], self.biases['conv3'],
                                    activation=tf.nn.relu)
        pool4 = tf.nn.max_pool(conv3,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        fatten5 = self.getFlattenLayer(pool4)

        if train:
            fatten5 = tf.nn.dropout(fatten5, keep_prob=1-dropout_ratio[0])

        fc6 = self.getDenseLayer(fatten5,
                                 self.weights['fc6'], self.biases['fc6'],
                                 activation=tf.nn.relu)

        if train:
            fc6 = tf.nn.dropout(fc6, keep_prob=1-dropout_ratio[1])

        fc7 = self.getDenseLayer(fc6,
                                 self.weights['fc7'], self.biases['fc7'],
                                 activation=tf.nn.relu)

        logits = self.getDenseLayer(fc7, self.weights['fc8'], self.biases['fc8'])

        y_ = tf.nn.softmax(logits)
        loss = tf.reduce_mean(
                 tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                         logits=logits))

        return (y_, loss)

    def getDenseLayer(self, input_layer, weight, bias, activation=None):
        x = tf.add(tf.matmul(input_layer, weight), bias)
        if activation:
            x = activation(x)
        return x

    def getConv2DLayer(self, nput_layer,
                       weight, bias,
                       strides=(1, 1), padding='VALID', activation=None):
        x = tf.add(
              tf.nn.conv2d(input_layer,
                           weight,
                           [1, strides[0], strides[1], 1],
                           padding=padding), bias)
        if activation:
            x = activation(x)
        return x

    def getFlattenLayer(self, input_layer):
        shape = input_layer.get_shape().as_list()
        n = 1
        for s in shape[1:]:
            n *= s
        x = tf.reshape(input_layer, [-1, n])
        return x

    def fit(self, X, y, epochs=10,
            validation_data=None, test_data=None, batch_size=None):
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
                    self.train_pictures: X[batch_index, :],
                    self.train_labels: y[batch_index],
                }
                _, loss = self.sess.run([self.train_op, self.loss],
                                        feed_dict=feed_dict)

                print('[%d/%d] loss = %.4f     ' % (N-len(index), N, loss), end='\r')

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
        return self.sess.run(self.new_y_, feed_dict={self.new_pictures: X})

    def evaluate(self, X, y):
        X = self._check_array(X)
        y = self._check_array(y)
        return self.sess.run(self.new_loss, feed_dict={self.new_pictures: X,
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

    train_img = np.reshape(train_data.images, [-1, 28, 28, 1])
    valid_img = np.reshape(valid_data.images, [-1, 28, 28, 1])
    test_img = np.reshape(test_data.images, [-1, 28, 28, 1])

    model = CNNLogisticClassification(
        shape_picture=[28, 28, 1],
        n_labels=10,
        learning_rate=0.07,
        dropout_ratio=[0.2, 0.1],
        alpha=0.1,
    )
    model.fit(
        X=train_img,
        y=train_data.labels,
        epochs=10,
        validation_data=(valid_img, valid_data.labels),
        test_data=(test_img, test_data.labels),
        batch_size=32,
    )
