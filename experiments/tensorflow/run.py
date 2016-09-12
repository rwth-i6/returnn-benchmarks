'''
A Bidirectional Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import h5py

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

BATCH_SIZE = 27
MAX_LEN = 250

class HDF5DATA(object):
  def one_hot(self, x):
    xs = x.reshape(x.shape[0] * x.shape[1], ) if len(x.shape) == 2 else x
    xs[xs==10429] = 0
    res = np.zeros(list(xs.shape) + [self.n_out],'int32')
    res[np.arange(xs.shape[0]), xs] = 1
    return res.reshape(x.shape[0],x.shape[1],self.n_out) if len(x.shape) == 2 else res

  def __init__(self, filename):  
    self.batches = []  
    h5 = h5py.File(filename, "r")
    lengths = h5["seqLengths"][...].T[0].tolist()
    xin = h5['inputs'][...]
    yin = h5['targets/data']['classes'][...]
    yin[yin==10429] = 0
    self.n_out = h5['targets/size'].attrs['classes']
    self.n_in = xin.shape[1]
    self.n_seqs = len(lengths)
    i = 0
    while i < len(lengths):
      end = min(i+BATCH_SIZE,len(lengths))
      batch_x = np.zeros((BATCH_SIZE, MAX_LEN, xin.shape[1]), 'float32')
      batch_y = np.zeros((BATCH_SIZE, MAX_LEN, self.n_out), 'int8')
      #batch_y = np.zeros((BATCH_SIZE, MAX_LEN), 'int32')
      batch_i = np.zeros((BATCH_SIZE, MAX_LEN), 'int8')
      for j in xrange(end-i):
        batch_x[j,:lengths[i+j]] = (xin[sum(lengths[:i+j]):sum(lengths[:i+j+1])])
        batch_y[j,:lengths[i+j]] = self.one_hot(yin[sum(lengths[:i+j]):sum(lengths[:i+j+1])])
        #batch_y[j,:lengths[i+j]] = yin[sum(lengths[:i+j]):sum(lengths[:i+j+1])]
        batch_i[j,:lengths[i+j]] = 1
      self.batches.append((batch_x,batch_y,batch_i,MAX_LEN)) #max(lengths[i:end])))
      i = end
    self.lengths = lengths
    h5.close()
    self.batch_idx = 0

  def next_batch(self):
    if self.batch_idx == len(self.batches):
      self.batch_idx = 0
      return None, None, None, None
    self.batch_idx += 1
    return self.batches[self.batch_idx-1]

train = HDF5DATA('data/train.0001')

# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = 27
display_step = 10

# Network Parameters
n_input = train.n_in # MNIST data input (img shape: 28*28)
n_steps = 250 # timesteps
n_hidden = 512 # hidden layer num of features
n_classes = train.n_out # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define lstm cells with tensorflow
    with tf.variable_scope("lstm1") as scope1:
        lstm_fw_cell_1 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_bw_cell_1 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs_1, _, _ = rnn.bidirectional_rnn(lstm_fw_cell_1, lstm_bw_cell_1, x, dtype=tf.float32)

    with tf.variable_scope("lstm2") as scope2:
        lstm_fw_cell_2 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_bw_cell_2 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs_2, _, _ = rnn.bidirectional_rnn(lstm_fw_cell_2, lstm_bw_cell_2, outputs_1, dtype=tf.float32)

    with tf.variable_scope("lstm3") as scope3:
        lstm_fw_cell_3 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_bw_cell_3 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs_3, _, _ = rnn.bidirectional_rnn(lstm_fw_cell_3, lstm_bw_cell_3, outputs_2, dtype=tf.float32)

    outputs = outputs_3
    outputs = tf.reshape(tf.concat(0, outputs), [MAX_LEN*BATCH_SIZE,n_hidden*2])
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights['out']) + biases['out']

with tf.device('/gpu:0'):
    pred = BiRNN(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    import time

    st = time.time()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        acc = 0
        loss = 0
        epoch = 0
        et = time.time()
        # Keep training until reach max iterations
        while epoch < 10:
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y, _, _ = train.next_batch()
            if batch_x is None:
                print("epoch " + str(epoch) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc) + ", Time Elapsed= " + str(et-time.time()))
                acc = 0
                loss = 0
                epoch += 1
                et = time.time()
                continue
            #print("epoch " + str(epoch) + ", Minibatch Loss= " + \
            #          "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #          "{:.5f}".format(acc) + ", Time Elapsed= " + str(et-time.time()))
            batch_y = batch_y.reshape((batch_y.shape[0]*batch_y.shape[1],batch_y.shape[2]))
            # Calculate batch accuracy
            acc += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss += sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            # Reshape data to get 28 seq of 28 elements
            #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        print("Optimization Finished!")
    print("Elapsed: %d" % (time.time()-st))
    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
