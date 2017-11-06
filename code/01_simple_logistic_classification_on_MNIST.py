#!/usr/local/bin/python3.6

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

class SimpleLogisticClassification(object):
    def __init__(self,n_features,n_labels,learning_rate=0.5):
        self.n_features = n_features
        self.n_labels = n_labels
        
        self.weights = None
        self.biases  = None
        
        self.graph = tf.Graph() # initialize new graph
        self.build(learning_rate) # building graph
        self.sess = tf.Session(graph=self.graph) # create session by the graph     
    
    def build(self,learning_rate):
        # Building Graph
        with self.graph.as_default():
            ### Input
            self.train_features = tf.placeholder(tf.float32, shape=(None,self.n_features))
            self.train_labels   = tf.placeholder(tf.int32  , shape=(None,self.n_labels))
            
            ### Optimalization
            # build neurel network structure and get their predictions and loss
            self.y_,self.loss = self.structure(features=self.train_features,
                                                        labels=self.train_labels)
            # define training operation
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            
            ### Prediction
            self.new_features = tf.placeholder(tf.float32, shape=(None,self.n_features))
            self.new_labels   = tf.placeholder(tf.int32  , shape=(None,self.n_labels))
            self.new_y_,self.new_loss = self.structure(features=self.new_features,
                                                       labels=self.new_labels,)
            
            ### Initialization
            self.init_op = tf.global_variables_initializer()
            
    def structure(self,features,labels):
        # build neurel network structure and return their predictions and loss
        ### Variable
        if (not self.weights) or (not self.biases):
            self.weights = {
                'fc1': tf.Variable(tf.truncated_normal( shape=(self.n_features,self.n_labels) )),
            }
            self.biases  = {
                'fc1': tf.Variable(tf.zeros( shape=(self.n_labels) )),
            } 
            
        ### Structure   
        # one fully connected layer
        logits = self.getDenseLayer(features,self.weights['fc1'],self.biases['fc1'])
        
        # predictions
        y_ = tf.nn.softmax(logits)
        
        # loss: softmax cross entropy
        loss = tf.reduce_mean(
                 tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))

        return (y_,loss)
    
    def getDenseLayer(self,input_layer,weight,bias,activation=None):
        # fully connected layer
        x = tf.add(tf.matmul(input_layer,weight),bias)
        if activation:
            x = activation(x)
        return x
    
    def fit(self,X,y,epochs=10,validation_data=None,test_data=None):
        X = self._check_array(X)
        y = self._check_array(y)
        
        self.sess.run(self.init_op)
        for epoch in range(epochs):
            print("Epoch %2d/%2d: "%(epoch+1,epochs))
            
            # fully gradient descent
            feed_dict = {self.train_features: X, self.train_labels: y}
            _ = self.sess.run(self.train_op, feed_dict=feed_dict)
            
            # evaluate at the end of this epoch
            y_ = self.predict(X)
            train_loss = self.evaluate(X,y)
            train_acc = self.accuracy(y_,y)
            msg = " loss = %8.4f, acc = %3.2f%%" % ( train_loss, train_acc*100 )
            
            if validation_data:
                val_loss = self.evaluate(validation_data[0],validation_data[1])
                val_acc = self.accuracy(self.predict(validation_data[0]),validation_data[1])
                msg += ", val_loss = %8.4f, val_acc = %3.2f%%" % ( val_loss, val_acc*100 )
            
            print(msg)
            
        if test_data:
            test_acc = self.accuracy(self.predict(test_data[0]),test_data[1])
            print("test_acc = %3.2f%%" % (test_acc*100))
            
    def accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])
    
    def predict(self,X):
        X = self._check_array(X)
        return self.sess.run(self.new_y_, feed_dict={self.new_features: X})
    
    def evaluate(self,X,y):
        X = self._check_array(X)
        y = self._check_array(y)
        return self.sess.run(self.new_loss, feed_dict={self.new_features: X, self.new_labels: y})
    
    def _check_array(self,ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape)==1: ndarray = np.reshape(ndarray,(1,ndarray.shape[0]))
        return ndarray
    
    
if __name__=="__main__":
    print("Extract MNIST Dataset ...")
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train_data = mnist.train
    valid_data = mnist.validation
    test_data = mnist.test

    model = SimpleLogisticClassification(n_features=28*28,
                                     n_labels=10,
                                     learning_rate= 0.5,)
    model.fit(X=train_data.images,
              y=train_data.labels,
              epochs=10,
              validation_data=(valid_data.images,valid_data.labels),
              test_data=(test_data.images,test_data.labels), )
