#!/bin/env/python
#
# audio_cnn.py
#
# trains a CNN on the audio data
#

import tensorflow as tf
import numpy as np

# custom libraries
from object_naming_cnn import ObjectNamingCNN
from basic_tfrecord_rw import *
from constants import *

# PARAMETERS
TRAIN_SET_SIZE = 0.1
EPOCHS = 30000
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
CLASSES = ["abort", "command", "correct", "incorrect", "prompt", "reward", "visual"]

###########################
# TF CNN functions
###########################
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, s=1):
	return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

def max_pool(k=2, s=2, x):
	'''pooling is performed over k x k blocks with dride size s'''
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')

def main():
    """trains and tests a CNN given a set of tfrecord files"""
    tfrecord_dir = '/home/assistive-robotics/object_naming_dataset/tfrecords/'  # where to get the data
    num_classes = len(CLASSES)

    # read the contents of the tfrecord_dir into a dataset
    filenames = os.listdir(tfrecord_dir)
    filename_queue = tf.train.string_input_producer(filenames)
    
    # start TF session
    with sess = tf.Session():
        context, sequence = parse_sequence_example(filename_queue)
        
        # placeholders
        # audio data placeholder
        x = tf.placeholder(tf.float32, [BATCH_SIZE, None,
                                aud_dtype["cmp_h"] * aud_dtype["cmp_w"] * aud_dtype["num_c"]],
								name="aud_placeholder")
        # sequence length placeholder
        seq_length_ph = tf.placeholder("int32", [BATCH_SIZE], name="seq_len_placeholder")
        # class label placeholder
        y_ = tf.placeholder(tf.int32, [None, num_classes])

        # first convolutional layer
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])



        # reshape audio input to 4d tensor


    # training and testing data
    self.X_train = None
    self.y_train = None
    self.X_test = None
    self.y_test = None


if __name__ == "__main__":
    main()