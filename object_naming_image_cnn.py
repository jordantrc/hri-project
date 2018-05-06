#!/bin/env/python
#
# audio_cnn.py
#
# trains a CNN on the audio data
#

import os
import random
import tensorflow as tf
import numpy as np

# custom libraries
from basic_tfrecord_rw import *
from constants import *

# PARAMETERS
TRAIN_SET_SIZE = 0.01
EPOCHS = 1000
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
DROPOUT = 0.5
RANDOM_SEED = 1
SHUFFLE = True



###########################
# TF CNN functions
###########################
# taken from: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# initialize weights with small amount of noise
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# give the neurons a slightly positive bias to avoid dead neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_batch(dataset, n, offset, num_features):
    '''returns a batch of samples'''
    X_batch = []
    y_batch = []

    # dataset format is data, label, length
    new_offset = offset
    while len(X_batch) < n:
        if new_offset >= len(dataset):
            # loop around
            new_offset = 0
        #x = dataset[new_offset][0].reshape(-1)
        # this is a single image
        x = dataset[new_offset][0].reshape(-1)[:num_features]
        y = dataset[new_offset][1]
        #print("x length: %s, x type: %s, x shape: %s\n y length: %s, y type: %s y shape: %s" % (len(x), type(x), x.shape, len(y), type(y), y.shape))
        X_batch.append(x)
        y_batch.append(y)
        new_offset += 1

    return X_batch, y_batch, new_offset


def get_tfrecord_batch(filenames, n, offset):
    '''obtains a batch of data from a set of tfrecord files
    returns a list of features and a list of labels and the new
    offset'''
    data_list = []

def load_tfrecord_data(directory, files, data, data_type):
    '''loads training data from tfrecord files'''
    data_list = []
    for f in files:
        print("EXTRACTING from %s" % (f))
        # todo fix this need the filename
        coord = tf.train.Coordinator()
        full_path = directory + f
        filename_queue = tf.train.string_input_producer([full_path])

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            # parse TFrecord
            context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
            threads = tf.train.start_queue_runners(coord=coord)

            seq_len = context_parsed["length"]  # sequence length
            label = context_parsed["label"]  # class labels
            label = tf.one_hot(label, depth=len(CLASSES))
            #print("===DEBUG===\nseq_len = %s\nlabel = %s\n===/DEBUG===" % (seq_len, label))

            data_s = tf.reshape(sequence_parsed[data], [-1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]])
            extract = tf.cast(data_s, tf.uint8)

            d, la, le = sess.run([extract, label, seq_len])

            coord.request_stop()
            coord.join(threads)

            #print("d shape: %s, type: %s, la shape: %s, type: %s, le shape: %s, type: %s" % (d.shape, 
            #                                                                                 type(d),
            #                                                                                 la.shape,
            #                                                                                 type(la),
            #                                                                                 le.shape,
            #                                                                                 type(le)))
            data_list.append([d, la, le])

    return data_list


def train_cnn(train_data, test_data, data_type):
    '''trains the cnn given a set of training tfrecords'''

    num_classes = len(CLASSES)

    coord = tf.train.Coordinator()

    # start TF session
    with tf.Session() as sess:
        # initializer

        threads = tf.train.start_queue_runners(coord=coord)
        num_features = data_type["cmp_h"] * data_type["cmp_w"] * data_type["num_c"]

        # placeholders
        #x = tf.placeholder(tf.float32, shape=[None, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]], name="x")
        x = tf.placeholder(tf.float32, shape=[None, num_features], name="x")
        keep_prob = tf.placeholder(tf.float32)
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true")
        y_true_cls = tf.argmax(y_true, axis=1)

        # reshape x to 4d tensor
        x_4d = tf.reshape(x, [-1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]])

        # create network
        # first convolutional layer
        # convolution , followed by max pooling
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        # reshape x_image with weight tensor, add the bias, apply ReLU function
        # finally max pool
        # max_pool_2x2 reduces image to 14x14
        h_conv1 = tf.nn.relu(conv2d(x_4d, W_conv1, b_conv1))
        h_pool1 = maxpool2d(h_conv1)
        
        # second convolutional layer
        # 64 features for each 5x5 patch
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        # max_pool_2x2 reduces image size to 7x7
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, b_conv2))
        h_pool2 = maxpool2d(h_conv2)
        h_pool2_shape = h_pool2.get_shape().as_list()

        # densely connected layer
        # fully-connected layer with 1024 neurons
        # reshape the tensor from the pooling layer into a batch of vectors
        # multiply by weight matrix, add a bias, and apply ReLU
        W_fc1 = weight_variable([h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3], 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # dropout - reduces overfitting
        # turned on during training, turned off during testing, controlled by the keep_prob placeholder
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        # readout layer
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])

        # logits layer - class prediction
        logits = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)

        y_pred = tf.nn.softmax(logits)
        y_pred_cls = tf.argmax(y_pred, axis=1)

        # loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        train_op = optimizer.minimize(loss_op)

        # evaluate
        correct_pred = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_true, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        offset = 0
        for i in range(EPOCHS):
            # obtain training batch
            train_image_batch, train_label_batch, offset = get_batch(train_data, BATCH_SIZE, offset, num_features)

            #print("len_batch: %s, x shape: %s, type: %s" % (len(train_image_batch), 
            #                                                train_image_batch[0].shape, 
            #                                                type(train_image_batch[0])))

            feed_dict_train = {x: train_image_batch, y_true: train_label_batch, keep_prob: DROPOUT}
            feed_dict_mini = {x: train_image_batch, y_true: train_label_batch, keep_prob: 1.0}
            sess.run(train_op, feed_dict=feed_dict_train)
            
            # report loss, accuracy every few hundred iterations
            if i % 200 == 0:
                loss, acc = sess.run([loss_op, accuracy], feed_dict=feed_dict_mini)

                print("epoch %d, mini-batch loss %g, training accuracy %g" % (i, loss, acc))

        # calculate accuracy for test set
        offset = 0
        test_image_batch, test_label_batch, offset = get_batch(test_data, BATCH_SIZE, offset, num_features)
        feed_dict_test = {x: test_image_batch, y_true: test_label_batch, keep_prob: 1.0}
        acc = sess.run(accuracy, feed_dict_test)
        print("test accuracy = %g" % (acc))

        saver = tf.train.Saver()
        save_path = saver.save(sess, "./checkpoints/model.ckpt")
        print("Model saved in path: %s" % save_path)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main():
    """trains and tests a CNN given a set of tfrecord files"""
    tfrecord_dir = '/home/assistive-robotics/object_naming_dataset/tfrecords/'  # where to get the data
    num_classes = len(CLASSES)
    data = "top_opt_raw"
    data_type = pnt_dtype

    # create train and test file sets
    filenames = os.listdir(tfrecord_dir)
    num_files = int(len(filenames) * TRAIN_SET_SIZE)
    train_filenames = [ filenames[i] for i in sorted(random.sample(xrange(len(filenames)), num_files)) ]
    test_filenames = [ f for f in filenames if f not in train_filenames ]

    # temporary for memory limitations
    test_filenames = [ test_filenames[i] for i in sorted(random.sample(xrange(len(test_filenames)), num_files)) ]

    print("training with %s tfrecords, testing with %s tfrecords" % (len(train_filenames), len(test_filenames)))

    print("Train files")
    train = load_tfrecord_data(tfrecord_dir, train_filenames, data, data_type)
    print("Test files")
    test = load_tfrecord_data(tfrecord_dir, test_filenames, data, data_type)
    #print("train head = %s" % (train[0:2]))

    if SHUFFLE:
        random.shuffle(train)
        random.shuffle(test)

    train_cnn(train, test, data_type)
    







if __name__ == "__main__":
    main()
