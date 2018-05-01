#!/bin/env/python
#
# audio_cnn.py
#
# trains a CNN on the audio data
#

import tensorflow as tf
import numpy as np

# custom libraries
from basic_tfrecord_rw import *
from constants import *

# PARAMETERS
TRAIN_SET_SIZE = 0.1
EPOCHS = 30000
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
CLASSES = ["abort", "command", "correct", "incorrect", "prompt", "reward", "visual"]
RANDOM_SEED = 1



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


def max_pool(x, k=2, s=2):
    '''pooling is performed over k x k blocks with stide size s'''
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')


def main():
    """trains and tests a CNN given a set of tfrecord files"""
    tfrecord_dir = '/home/assistive-robotics/object_naming_dataset/tfrecords/'  # where to get the data
    num_classes = len(CLASSES)

    # create train and test file sets
    filenames = os.listdir(tfrecord_dir)
    train_filenames = [filenames[i] for i in sorted(random.sample(xrange(len(filenames),
                                                                  int(len(filenames) * TRAIN_SET_SIZE))))]
    test_filenames = [ f for f in filenames if f not in train_filenames ]
    random.shuffle(train_filenames)

    train_queue = tf.train.string_input_producer(train_filenames)
    coord = tf.train.Coordinator()

    # start TF session
    with tf.Session() as sess:
        context, sequence = parse_sequence_example(train_queue)
        threads = tf.train.start_queue_runners(coord=coord)
        
        # placeholders
        # audio data placeholder
        x = tf.placeholder(tf.float32, [BATCH_SIZE, None,
                           aud_dtype["cmp_h"] * aud_dtype["cmp_w"] * aud_dtype["num_c"]],
                           name="aud_placeholder")
        # sequence length placeholder
        seq_length_ph = tf.placeholder("int32", [BATCH_SIZE], name="seq_len_placeholder")
        # class label placeholder
        y_ = tf.placeholder(tf.int32, [None, num_classes])

        # first convolutional layer - first initialize with random weights
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool(h_conv1)  # 2x2 blocks, stride size 2

        # second convolutional layer
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool(h_conv2)

        # densely connected layer
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout - reduced overfitting
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        true_positives = tf.metrics.true_positives(y_, y_conv)
        cross_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(EPOCHS):
            sess.run(tf.global_variables_initializer())






if __name__ == "__main__":
    main()
