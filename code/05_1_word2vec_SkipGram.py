#!/usr/local/bin/python3.6

from __future__ import print_function
from __future__ import generators
import collections
import os
import zipfile
from six.moves.urllib.request import urlretrieve
import tensorflow as tf
import numpy as np
import random
import math
import time

VOCABULARY_SIZE = 100000

def maybe_download(url,filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
  
def build_dataset(words,vocabulary_size=VOCABULARY_SIZE):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary


def skip_gram_batch_generator(data,batch_size,num_skips,skip_window):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    
    # initialization
    data_index = 0
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
    # generate
    k = 0
    while True:
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ target ]
        for _ in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[k] = buffer[skip_window]
            labels[k, 0] = buffer[target]
            k += 1
            
        # Recycle 
        if data_index == len(data): data_index = 0
        
        # scan data
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
        # Enough num to output
        if k == batch_size:
            k = 0
            yield (batch.copy(), labels.copy())

class SkipGram(object):
    def __init__(self,n_vocabulary,n_embedding,reverse_dictionary,learning_rate=1.0):
        self.n_vocabulary = n_vocabulary
        self.n_embedding = n_embedding
        self.reverse_dictionary = reverse_dictionary
        
        self.weights = None
        self.biases = None
        
        self.graph = tf.Graph() # initialize new grap
        self.build(learning_rate) # building graph
        self.sess = tf.Session(graph=self.graph) # create session by the graph 
        
    def build(self,learning_rate):
        with self.graph.as_default():
            ### Input
            self.train_dataset = tf.placeholder(tf.int32, shape=[None])
            self.train_labels  = tf.placeholder(tf.int32, shape=[None, 1])

            ### Optimalization
            # build neurel network structure and get their loss
            self.loss = self.structure( dataset=self.train_dataset,
                                        labels=self.train_labels,
                                      )
            
            # normalize embeddings
            self.norm = tf.sqrt(
                          tf.reduce_sum(
                            tf.square(self.weights['embeddings']), 1, keep_dims=True))
            self.normalized_embeddings = self.weights['embeddings'] / self.norm
            
            # define training operation
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            
            ### Prediction
            self.new_dataset = tf.placeholder(tf.int32, shape=[None])
            self.new_labels  = tf.placeholder(tf.int32, shape=[None, 1])
            self.new_loss = self.structure( dataset=self.new_dataset,
                                            labels=self.new_labels,
                                          )
            
            # similarity
            self.new_embed = tf.nn.embedding_lookup(
                               self.normalized_embeddings, self.new_dataset)
            self.new_similarity = tf.matmul(self.new_embed, 
                                            tf.transpose(self.normalized_embeddings))
            
            ### Initialization
            self.init_op = tf.global_variables_initializer()  
    
    def structure(self,dataset,labels):
        ### Variable
        if (not self.weights) and (not self.biases):
            self.weights = {
                'embeddings': tf.Variable(
                                tf.random_uniform([self.n_vocabulary, self.n_embedding], 
                                                  -1.0, 1.0)),
                'softmax': tf.Variable(
                             tf.truncated_normal([self.n_vocabulary, self.n_embedding],
                               stddev=1.0 / math.sqrt(self.n_embedding)))
            }
            self.biases = {
                'softmax': tf.Variable(tf.zeros([self.n_vocabulary]))
            }
                              

        ### Structure
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(self.weights['embeddings'], dataset)
        
        # Compute the softmax loss, using a sample of the negative labels each time.
        num_softmax_sampled = 64
        
        loss = tf.reduce_mean(
                 tf.nn.sampled_softmax_loss(weights=self.weights['softmax'], 
                                            biases=self.biases['softmax'], 
                                            inputs=embed,
                                            labels=labels, 
                                            num_sampled=num_softmax_sampled, 
                                            num_classes=self.n_vocabulary))

        return loss
    
    
    def initialize(self):
        self.weights = None
        self.biases = None
        self.sess.run(self.init_op)
    
    def online_fit(self,X,Y):      
        feed_dict = {self.train_dataset: X,
                     self.train_labels: Y}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        
        return loss
    
    def nearest_words(self,X,top_nearest):
        similarity = self.sess.run(self.new_similarity,
                                   feed_dict={self.new_dataset: X})
        X_size = X.shape[0]
        
        valid_words = []
        nearests = []
        for i in range(X_size):
            valid_word = self.find_word(X[i])
            valid_words.append(valid_word)    
            
            # select highest similarity word
            nearest = (-similarity[i, :]).argsort()[1:top_nearest+1]  
            nearests.append(list(map(lambda x:self.find_word(x),nearest)))
            
        return (valid_words,np.array(nearests))
    
    def evaluate(self,X,Y):
        return self.sess.run(self.new_loss, feed_dict={self.new_dataset: X,
                                                       self.new_labels: Y})
    
    def embedding_matrix(self):
        return self.sess.run(self.normalized_embeddings)
    
    def find_word(self,index):
        return self.reverse_dictionary[index]


def main():
    ### download and load data
    print("Downloading text8.zip")
    filename = maybe_download('http://mattmahoney.net/dc/text8.zip','./text8.zip', 31344016)
    
    print("=====")
    words = read_data(filename)
    print('Data size %d' % len(words))
    print('First 10 words: {}'.format(words[:10]))
    
    print("=====")
    data, count, dictionary, reverse_dictionary = build_dataset(words,
                                                                vocabulary_size=VOCABULARY_SIZE)
    del words  # Hint to reduce memory.
    
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    
    ### train model
    # build skip-gram batch generator
    batch_generator = skip_gram_batch_generator(data=data,
                                                batch_size=128,
                                                num_skips=2,
                                                skip_window=1)
    
    # build skip-gram model
    model_SkipGram = SkipGram(n_vocabulary=VOCABULARY_SIZE,
                              n_embedding=100,
                              reverse_dictionary=reverse_dictionary,
                              learning_rate=1.0)
    # initial model
    model_SkipGram.initialize()
    
    # online training
    epochs = 50
    num_batchs_in_epoch = 5000
    
    for epoch in range(epochs):
        start_time = time.time()
        avg_loss = 0
        for _ in range(num_batchs_in_epoch):
            batch, labels = next(batch_generator)
            loss = model_SkipGram.online_fit(X=batch,
                                             Y=labels)
            avg_loss += loss
        avg_loss = avg_loss / num_batchs_in_epoch
        print("Epoch %d/%d: %ds loss = %9.4f" % ( epoch+1, epochs, time.time()-start_time,
                                                       avg_loss ))
    
    ### nearest words
    valid_words_index = np.array([10,20,30,40,50,210,239,392,396])
    
    valid_words, nearests = model_SkipGram.nearest_words(X=valid_words_index,top_nearest=8)
    for i in range(len(valid_words)):
        print("Nearest to '{}': ".format(valid_words[i]),nearests[i])
    

    ### visualization
    from matplotlib import pylab
    from sklearn.manifold import TSNE
    
    def plot(embeddings, labels):
        assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(15,15))  # in inches
        for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            pylab.scatter(x, y, color="blue")
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
        pylab.show()
    
    visualization_words = 800
    # transform embeddings to 2D by t-SNE
    embed = model_SkipGram.embedding_matrix()[1:visualization_words+1,:]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    two_d_embed = tsne.fit_transform(embed)
    # list labels
    words = [model_SkipGram.reverse_dictionary[i] for i in range(1, visualization_words+1)]
    # plot
    plot(two_d_embed,words)

if __name__ == "__main__":
    main()
