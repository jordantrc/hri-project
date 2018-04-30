#!/bin/env/python
#
# Library for the Object Naming dataset CNNs
#

import tensorflow as tf
import os

from basic_tfrecord_rw import *
from constants import *


class ObjectNamingCNN:

    def __init__(self, tfrecord_dir, iterations, train_set_size, classes):
        self.tfrecord_dir = tfrecord_dir  # where to get the data
        # CNN parameters
        self.iterations = iterations
        self.train_set_size = train_set_size
        self.test_set_size = 1.0 - train_set_size
        self.classes = classes

        # read the contents of the tfrecord_dir into a dataset
        filenames = os.listdir(self.tfrecord_dir)
        self.filename_queue = tf.train.string_input_producer(filenames)
        self.tfsession = None

        # training and testing data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self, num_features, placeholders, dataset_name):
    	self.tfsession = tf.Session()
        self.context, self.sequence = parse_sequence_example(self.filename_queue)

        

    def train_cnn(self):
    	self.tfsession.run(tf.local_variables_initializer())

    	# create the data placeholders for tensorflow
        num_classes = len(classes)
        x = tf.placeholder(tf.float32, [None, num_features])
        y_ = tf.placeholder(tf.int32, [None, num_classes])


    def test_cnn(self):


    ###########################
    # TF CNN functions - static
    ###########################
    def weight_variable(shape):
    	initial = tf.truncated_normal(shape, stddev=0.1)
    	return tf.Variable(initial)

    def bias_variable(shape):
    	initial = tf.constanct(0.1, shape=shape)
    	return tf.Variable(initial)

    def conv2d(x, W):
    	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_nxn(n, x):
    	'''pooling is performed over n x n blocks'''
    	return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')


