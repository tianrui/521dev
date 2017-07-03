# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from progressbar import ETA, Bar, Percentage, ProgressBar
import time
import pdb
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def fc_layer(in_tensor, hidden_size, decay_coef):
    W = weight_variable([in_tensor.get_shape().as_list()[1], hidden_size], decay_coef)
    b = weight_variable([hidden_size], decay_coef)
    return tf.matmul(in_tensor, W) + b

def buildfc(x, decay_coef):
    keep_prob = tf.placeholder(tf.float32)

    x_flat = tf.reshape(x, [-1, 28 * 28])
    h1 = tf.nn.relu(fc_layer(x_flat, 512, decay_coef))
    h1_drop = tf.nn.dropout(h1, keep_prob)

    h2 = tf.nn.relu(fc_layer(h1_drop, 128, decay_coef))
    W_3 = weight_variable([128, 10], decay_coef)
    b_3 = weight_variable([10], decay_coef)
    y = tf.matmul(h2, W_3) + b_3

    return y, keep_prob
    

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  # W_conv2 = weight_variable([5, 5, 32, 64])
  # b_conv2 = bias_variable([64])
  # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # # Second pooling layer.
  # h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([14 * 14 * 32, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool1, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, decay_coef):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  var = tf.Variable(initial)
  if decay_coef is not None:
      decay = tf.multiply(tf.nn.l2_loss(var), decay_coef, name='weight_loss')
      tf.add_to_collection('losses', decay)
  return var


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def part1(_):
    batch_size = 50
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_fc, keep_prob = buildnet(x, 3e-4)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fc))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_fc, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Config session for memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement=True
  
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
  
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  
            if i % 1000 == 0:
                acc = 0
                for i in range(int(mnist.test._num_examples / batch_size)):
                    xtest, ytest = mnist.test.next_batch(batch_size)
                    acc += accuracy.eval(feed_dict={x: xtest, y_: ytest, keep_prob: 1.0})
  
                acc /= int(mnist.test._num_examples / batch_size)
                print('test accuracy %g' % acc)
  
        #print('test accuracy %g' % accuracy.eval(feed_dict={
        #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
  
        end = time.time()
        print('Time elapsed: %f' % (end-start))

def notmnist_fc(_):
  # Define hyperparameters
  batch_size = 32
  epochs = 25
  train_size = 15000
  valid_size = 1000
  decay_coef = 3e-4
  # Define logging arrays
  train_ce = np.zeros(epochs)
  train_acc = np.zeros(epochs)
  valid_ce = np.zeros(epochs)
  valid_acc = np.zeros(epochs)

  # Import data
  data = np.load('./notMNIST.npz', 'r')
  notmnist_x, notmnist_y = data["images"], data["labels"]
  randIdx = np.arange(len(notmnist_x))
  np.random.seed(521)
  np.random.shuffle(randIdx)
  notmnist_x = notmnist_x[randIdx]/255.
  notmnist_y = notmnist_y[randIdx]
  notmnist_x = np.reshape(notmnist_x, (-1, 28, 28))
  notmnist_y = np.eye(10)[notmnist_y]
  notmnist_xtrain, notmnist_ytrain = notmnist_x[:train_size], notmnist_y[:train_size]
  notmnist_xvalid, notmnist_yvalid = notmnist_x[train_size:train_size+valid_size], notmnist_y[train_size:train_size+valid_size]
  notmnist_xtest, notmnist_ytest = notmnist_x[train_size+valid_size:], notmnist_y[train_size+valid_size:]

  # Create the model
  x = tf.placeholder(tf.float32, [None, 28, 28])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the fc net
  y_pred, keep_prob = buildfc(x, decay_coef)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
  tf.add_to_collection('losses', cross_entropy)
  loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
  correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # Config session for memory
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  #config.gpu_options.per_process_gpu_memory_fraction = 0.5
  config.log_device_placement=True

  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    start = time.time()

    for j in range(epochs):
      for i in range((int)(train_size/batch_size)):
          batch = [notmnist_xtrain[(i*batch_size)%len(notmnist_xtrain): ((i+1)*batch_size)%len(notmnist_xtrain)], notmnist_ytrain[(i*batch_size)%len(notmnist_ytrain): (i+1)*batch_size]]
          if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            train_ce[j] = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            train_acc[j] = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
          train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      acc = 0
      ce = 0
      for i in range(int(notmnist_xvalid.shape[0] / batch_size)):
        xtest, ytest = [notmnist_xvalid[(i*batch_size)%len(notmnist_xvalid): ((i+1)*batch_size)%len(notmnist_xvalid)], notmnist_yvalid[(i*batch_size)%len(notmnist_yvalid): ((i+1)*batch_size)%len(notmnist_yvalid)]]
        acc += accuracy.eval(feed_dict={x: xtest, y_: ytest, keep_prob: 1.0})
        ce += cross_entropy.eval(feed_dict={x: xtest, y_: ytest, keep_prob: 1.0})

      acc /= int(notmnist_xvalid.shape[0] / batch_size)
      ce /= int(notmnist_xvalid.shape[0] / batch_size)

      print('validation accuracy %g' % acc)
      print('validation cross-entropy %g' % ce)
      valid_ce[j] = ce
      valid_acc[j] = acc


    #print('test accuracy %g' % accuracy.eval(feed_dict={
    #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    end = time.time()
    print('Time elapsed: %f' % (end-start))

    # plot figures
    plt.plot(train_acc, label='training accuracy')
    plt.plot(valid_acc, label='validation accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.savefig('./part1_acc.jpg')
    plt.plot(train_ce, label='training CE')
    plt.plot(valid_ce, label='validation CE')
    plt.legend()
    plt.show()
    plt.savefig('./part1_ce.jpg')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  #tf.app.run(main=part1, argv=[sys.argv[0]] + unparsed)
  tf.app.run(main=notmnist_fc, argv=[sys.argv[0]] + unparsed)
