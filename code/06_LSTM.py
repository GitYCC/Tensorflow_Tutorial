#!/usr/local/bin/python3.6

from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import time

LETTER_SIZE = len(string.ascii_lowercase) + 1 # [a-z] + ' '
FIRST_LETTER_ASCII = ord(string.ascii_lowercase[0])


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
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - FIRST_LETTER_ASCII + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0
    
def id2char(dictid):
    if dictid > 0:
        return chr(dictid + FIRST_LETTER_ASCII - 1)
    else:
        return ' '


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

def rnn_batch_generator(text, batch_size, num_unrollings):
    text_size = len(text)
    
    ### initialization
    segment = text_size // batch_size
    cursors = [ offset * segment for offset in range(batch_size)]
    
    batches = []
    batch_initial = np.zeros(shape=(batch_size, LETTER_SIZE), dtype=np.float)
    for i in range(batch_size):
        cursor = cursors[i]
        id_ = char2id(text[cursor])
        batch_initial[i][id_] = 1.0
        
        #move cursor
        cursors[i] = (cursors[i] + 1) % text_size
        
    batches.append(batch_initial) 

    ### generate loop
    while True:
        batches = [ batches[-1] ]
        for _ in range(num_unrollings):
            batch = np.zeros(shape=(batch_size, LETTER_SIZE), dtype=np.float)
            for i in range(batch_size):
                cursor = cursors[i]
                id_ = char2id(text[cursor])
                batch[i][id_] = 1.0
                
                #move cursor
                cursors[i] = (cursors[i] + 1) % text_size
            batches.append(batch)
            
        yield batches  # [last batch of previous batches] + [unrollings]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, LETTER_SIZE], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

class LSTM(object):
    def __init__(self,n_unrollings,n_memory,n_train_batch,learning_rate=1.0):
        self.n_unrollings = n_unrollings
        self.n_memory = n_memory

        self.weights = None
        self.biases = None
        self.saved = None
        
        self.graph = tf.Graph() # initialize new grap
        self.build(learning_rate,n_train_batch) # building graph
        self.sess = tf.Session(graph=self.graph) # create session by the graph 
        
    def build(self,learning_rate,n_train_batch):
        with self.graph.as_default():
            ### Input      
            self.train_data = list()
            for _ in range(self.n_unrollings + 1):
                self.train_data.append(
                    tf.placeholder(tf.float32, shape=[n_train_batch,LETTER_SIZE]))
            self.train_inputs = self.train_data[:self.n_unrollings]
            self.train_labels = self.train_data[1:]  # labels are inputs shifted by one time step.
    
    
            ### Optimalization
            # build neurel network structure and get their loss
            self.y_, self.loss = self.structure( inputs=self.train_inputs,
                                                 labels=self.train_labels,
                                                 n_batch=n_train_batch,
                                               )
            
            # define training operation
            
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            
            # gradient clipping
            gradients, v = zip(*self.optimizer.compute_gradients(self.loss)) # output gradients one by one
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25) # clip gradient
            self.train_op = self.optimizer.apply_gradients(zip(gradients, v)) # apply clipped gradients
            
            
            ### Sampling and validation eval: batch 1, no unrolling.
            self.sample_input = tf.placeholder(tf.float32, shape=[1, LETTER_SIZE])
            
            saved_sample_output = tf.Variable(tf.zeros([1, self.n_memory]))
            saved_sample_state = tf.Variable(tf.zeros([1, self.n_memory]))
            self.reset_sample_state = tf.group(     # reset sample state operator
                saved_sample_output.assign(tf.zeros([1, self.n_memory])),
                saved_sample_state.assign(tf.zeros([1, self.n_memory])))
            
            sample_output, sample_state = self.lstm_cell(
                self.sample_input, saved_sample_output, saved_sample_state)
            with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                          saved_sample_state.assign(sample_state)]):
                # use tf.control_dependencies to make sure "saving" before "prediction"
                
                self.sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, 
                                                                  self.weights['classifier'], 
                                                                  self.biases['classifier']))
            
            ### Initialization
            self.init_op = tf.global_variables_initializer()  
    
    def lstm_cell(self,i,o,state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        ## Build Input Gate
        ix = self.weights['input_gate_i']
        im = self.weights['input_gate_o']
        ib = self.biases['input_gate']
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        ## Build Forget Gate
        fx = self.weights['forget_gate_i']
        fm = self.weights['forget_gate_o']
        fb = self.biases['forget_gate']        
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        ## Memory
        cx = self.weights['memory_i']
        cm = self.weights['memory_o']
        cb = self.biases['memory']
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        ## Update State
        state = forget_gate * state + input_gate * tf.tanh(update)
        ## Build Output Gate        
        ox = self.weights['output_gate_i']
        om = self.weights['output_gate_o']
        ob = self.biases['output_gate']
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        ## Ouput
        output = output_gate * tf.tanh(state)
        
        return output, state
    
    def structure(self,inputs,labels,n_batch):
        ### Variable
        if (not self.weights) or (not self.biases) or (not self.saved):
            self.weights = {
              'input_gate_i': tf.Variable(tf.truncated_normal([LETTER_SIZE,self.n_memory], -0.1, 0.1)),
              'input_gate_o': tf.Variable(tf.truncated_normal([self.n_memory,self.n_memory], -0.1, 0.1)),
              'forget_gate_i': tf.Variable(tf.truncated_normal([LETTER_SIZE,self.n_memory], -0.1, 0.1)),
              'forget_gate_o': tf.Variable(tf.truncated_normal([self.n_memory,self.n_memory], -0.1, 0.1)),
              'output_gate_i': tf.Variable(tf.truncated_normal([LETTER_SIZE,self.n_memory], -0.1, 0.1)),
              'output_gate_o': tf.Variable(tf.truncated_normal([self.n_memory,self.n_memory], -0.1, 0.1)),
              'memory_i': tf.Variable(tf.truncated_normal([LETTER_SIZE,self.n_memory], -0.1, 0.1)),
              'memory_o': tf.Variable(tf.truncated_normal([self.n_memory,self.n_memory], -0.1, 0.1)),
              'classifier': tf.Variable(tf.truncated_normal([self.n_memory, LETTER_SIZE], -0.1, 0.1)),

            }
            self.biases = {
              'input_gate': tf.Variable(tf.zeros([1, self.n_memory])),
              'forget_gate': tf.Variable(tf.zeros([1, self.n_memory])),
              'output_gate': tf.Variable(tf.zeros([1, self.n_memory])),
              'memory': tf.Variable(tf.zeros([1, self.n_memory])),
              'classifier': tf.Variable(tf.zeros([LETTER_SIZE])),
            }
            
        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([n_batch, self.n_memory]), trainable=False)
        saved_state = tf.Variable(tf.zeros([n_batch, self.n_memory]), trainable=False)
                              
        ### Structure
        # Unrolled LSTM loop.
        outputs = list()
        output = saved_output
        state = saved_state
        for input_ in inputs:
            output, state = self.lstm_cell(input_, output, state)
            outputs.append(output)
        
        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output),
                                      saved_state.assign(state)]):
            # use tf.control_dependencies to make sure "saving" before "calculating loss"
            
            # Classifier
            logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), 
                                     self.weights['classifier'], 
                                     self.biases['classifier'])
            y_ = tf.nn.softmax(logits)
            loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.concat(labels, 0), logits=logits))
            
        return y_, loss
    
        
    def initialize(self):
        self.weights = None
        self.biases = None
        self.sess.run(self.init_op)
    
    def online_fit(self,X):      
        feed_dict = dict()
        for i in range(self.n_unrollings + 1):
            feed_dict[self.train_data[i]] = X[i]
            
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)    
        return loss
    
    def perplexity(self,X):
        sum_logprob = 0
        sample_size = len(X)-1
        batch_size = X[0].shape[0]
        
        for i in range(batch_size):
            self.sess.run(self.reset_sample_state)
            for j in range(sample_size):
                sample_input = np.reshape(X[j][i],newshape=(1,-1))
                sample_label = np.reshape(X[j+1][i],newshape=(1,-1))
                predictions = self.sess.run(self.sample_prediction,
                                            feed_dict={self.sample_input: sample_input})
                sum_logprob += logprob(predictions, sample_label)
        perplexity = float(np.exp(sum_logprob / batch_size / sample_size))
        return perplexity
    
    def generate(self,c,len_generate):
        feed = np.array([[1 if id2char(i)==c else 0 for i in range(LETTER_SIZE)]])
        sentence = characters(feed)[0]
        self.sess.run(self.reset_sample_state)
        for _ in range(len_generate):
            prediction = self.sess.run(self.sample_prediction,feed_dict={self.sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
        return sentence
 

def main():
    ### download and load data
    print("Downloading text8.zip")
    filename = maybe_download('http://mattmahoney.net/dc/text8.zip','./text8.zip', 31344016)
    
    print("=====")
    text = read_data(filename)
    print('Data size %d letters' % len(text))
    
    print("=====")
    valid_size = 1000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    train_size = len(train_text)
    print('Train Dataset: size:',train_size,'letters,\n  first 64:',train_text[:64])
    print('Validation Dataset: size:',valid_size,'letters,\n  first 64:',valid_text[:64])
    
        
    # build training batch generator
    batch_size=64
    num_unrollings=10

    batch_generator = rnn_batch_generator(text=train_text,
                                          batch_size=batch_size,
                                          num_unrollings=num_unrollings)
    
    # build validation data
    valid_batches = rnn_batch_generator(text=valid_text, 
                                        batch_size=1, 
                                        num_unrollings=1)
    
    valid_data = [np.array(next(valid_batches)) for _ in range(valid_size)]
    
    # build LSTM model
    model_LSTM = LSTM(n_unrollings=num_unrollings,
                      n_memory=128,
                      n_train_batch=batch_size,
                      learning_rate=0.9)
    # initial model
    model_LSTM.initialize()
    
    # online training
    epochs = 30
    num_batchs_in_epoch = 5000
    valid_freq = 5
    
    for epoch in range(epochs):
        start_time = time.time()
        avg_loss = 0
        for _ in range(num_batchs_in_epoch):
            batch = next(batch_generator)
            loss = model_LSTM.online_fit(X=batch)
            avg_loss += loss
            
        avg_loss = avg_loss / num_batchs_in_epoch
        
        train_perplexity = model_LSTM.perplexity(batch)
        print("Epoch %d/%d: %ds loss = %6.4f, perplexity = %6.4f" % ( epoch+1, epochs, time.time()-start_time,
                                                       avg_loss, train_perplexity))
        
        if (epoch+1) % valid_freq == 0:
            print("")
            print("=============== Validation ===============")
            print("validation perplexity = %6.4f" % (model_LSTM.perplexity(valid_data)))
            print("Generate From 'a':  ",model_LSTM.generate(c='a',len_generate=50))
            print("Generate From 'h':  ",model_LSTM.generate(c='h',len_generate=50))
            print("Generate From 'm':  ",model_LSTM.generate(c='m',len_generate=50))
            print("==========================================")
            print("")

if __name__ == "__main__":
    main()
