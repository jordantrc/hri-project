#!/bin/env/python
#
# object_naming_cnn.py
#
# trains a CNN on the audio data
#

import cv2
import csv
import itertools
import os
import random
import time
import tensorflow as tf
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# custom libraries
from basic_tfrecord_rw import *
from constants import *

# PARAMETERS
TRAIN_SET_SIZE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EPOCHS = 5000
BATCH_SIZE = 20
LEARNING_RATE = 1e-4
DROPOUT = 0.5
RANDOM_SEED = 1
SHUFFLE = True
IMAGE_STACK_SIZE = 10


class SampleGenerator():
    '''Sample Generator class, yields batches of samples to save memory'''

    def __init__(self, directory, files, batch_size, data_name, data_type, num_images, shuffle=True):
        self.directory = directory
        self.files = files
        self.num_files = len(files)
        self.batch_size = batch_size
        self.data_name = data_name
        self.data_type = data_type
        self.num_features = self.data_type['cmp_h'] * self.data_type['cmp_w'] * self.data_type['num_c']
        self.num_images = num_images
        self.shuffle = shuffle
        self.index = 0

        if self.shuffle:
            random.shuffle(self.files)

    def get_sample(self):
        X_batch = []
        y_batch = []

        for i in range(self.batch_size):
            if self.index >= self.num_files:
                self.index = 0
            file = self.files[self.index]
            print("EXTRACTING from %s" % (file))
            # todo fix this need the filename
            full_path = self.directory + file

            data, la, le = self.get_sample_from_tfrecord(full_path, self.data_name, self.data_type)

            X_batch.append(data)
            y_batch.append(la)

            self.index += 1

        return X_batch, y_batch

    def stack_data(self, data, data_type, n, length):
        '''returns a stack of data'''
        data = data.reshape(-1)

        frames = []
        stack_frame_indices = []
        all_indices = []

        for i in range(0, length):
            all_indices.append(i)

        # create a list of frame indices
        for i in range(n):
            stack_frame_indices.append(random.choice(all_indices))

        for i in stack_frame_indices:
            frame_start = i * self.num_features
            frame_end = frame_start + self.num_features
            frame = data[frame_start:frame_end]
            frames.append(frame)

        # now frames contains a num_files frames, average these
        accum = [0] * self.num_features
        for i, fr in enumerate(frames):
            for j, f in enumerate(fr):
                accum[j] += f

        # now average
        avg = []
        for a in accum:
            avg.append(int(a / self.num_files))

        return avg

    @staticmethod
    def get_sample_from_tfrecord(file_path, data_name, data_type):
        num_features = data_type["cmp_h"] * data_type["cmp_w"] * data_type["num_c"]
        img_h = 64
        img_w = 64
        coord = tf.train.Coordinator()
        filename_queue = tf.train.string_input_producer([file_path])

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            # parse TFrecord
            context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
            threads = tf.train.start_queue_runners(coord=coord)

            seq_len = context_parsed["length"]  # sequence length
            label = context_parsed["label"]  # class labels
            label = tf.one_hot(label, depth=len(CLASSES))
            #print("===DEBUG===\nseq_len = %s\nlabel = %s\n===/DEBUG===" % (seq_len, label))

            data_s = tf.reshape(sequence_parsed[data_name], [-1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]])
            extract = tf.cast(data_s, tf.uint8)

            d, la, le = sess.run([extract, label, seq_len])

            coord.request_stop()
            coord.join(threads)

            # average random sample of frames
            # resize image to img_h * img_w
            d = d.reshape(-1)

            frames = []
            stack_frame_indices = []
            all_indices = []

            for i in range(0, le):
                all_indices.append(i)

            # create a list of frame indices
            # sampling with replacement
            for i in range(IMAGE_STACK_SIZE):
                stack_frame_indices.append(random.choice(all_indices))

            for i in stack_frame_indices:
                frame_start = i * num_features
                frame_end = frame_start + num_features
                frame = d[frame_start:frame_end]

                # resize the frame
                frame.reshape((data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]))

                frame = cv2.resize(frame, (img_h, img_w), interpolation=cv2.INTER_CUBIC)
                frame = frame.reshape(-1)
                #print("frame type = %s, shape = %s" % (type(frame), frame.shape))
                frames.append(frame)

            # reset size of num_features
            new_num_features = img_h * img_w * data_type["num_c"]

            # now frames contains a num_files frames, average these
            accum = [0] * new_num_features
            for i, fra in enumerate(frames):
                for j, fr in enumerate(fra):
                    accum[j] += fr

            # now average and normalize
            avg = []
            for a in accum:
                # average
                #print("a: type %s" % (type(a)))
                val = int(a / IMAGE_STACK_SIZE)
                # normalize
                val = float(val) / 255.0
                avg.append(val)

            # convert to numpy array
            avg = np.array(avg)

        sess.close()
        tf.reset_default_graph()

        return [avg, la, le]



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


def get_batch(dataset, n, offset):
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
        x = dataset[new_offset][0]
        y = dataset[new_offset][1]
        #print("x length: %s, x type: %s, x shape: %s\n y length: %s, y type: %s y shape: %s" % (len(x), type(x), x.shape, len(y), type(y), y.shape))
        X_batch.append(x)
        y_batch.append(y)
        new_offset += 1

    return X_batch, y_batch, new_offset


def balanced_files(files, num_files):
    '''returns a list of files balanced across classes'''
    files_by_class = {}
    file_list = []

    # generate a list of files for each class
    for c in CLASSES:
        files_by_class[c] = []
        for f in files:
            if c in f:
                files_by_class[c].append(f)

    # just get a random set if num_files less than number of classes
    if num_files < len(CLASSES):
        file_list = [ files[i] for i in sorted(random.sample(xrange(len(files)), num_files)) ]
    else:
        i = 0
        while len(file_list) < num_files:
            c = CLASSES[i]
            f = random.choice(files_by_class[c])
            file_list.append(f)
            i += 1
            if i >= len(CLASSES):
                i = 0

    return file_list


def load_tfrecord_data(directory, files, data, data_type):
    '''loads training data from tfrecord files'''
    data_list = []
    num_files = len(files)

    for i, f in enumerate(files):
        print("%s/%s EXTRACTING from %s" % (i, num_files, f))
        # todo fix this need the filename
        full_path = directory + f
        data_list.append(SampleGenerator.get_sample_from_tfrecord(full_path, data, data_type))

    return data_list


def tf_confusion_matrix(predictions, labels, classes):
    """
    produces and returns a confusion matrix given the predictions generated by
    tensorflow (in one-hot format), and string labels.
    """
    #print("pred = %s, type = %s, labels = %s, type = %s, classes = %s, type = %s" % (predictions, type(predictions), labels, type(labels), classes, type(classes)))

    y_true = []
    y_pred = []

    for p in predictions:
        y_true.append(classes[p])

    for l in labels:
        index = np.argmax(l)
        y_pred.append(classes[index])

    cm = metrics.confusion_matrix(y_true, y_pred, classes)

    return cm


def plot_confusion_matrix(cm, classes, filename,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() * 0.73
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.4f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(filename)
    plt.gcf().clear()
    plt.cla()
    plt.clf()
    plt.close()


def train_cnn(train, test, model_name, data_type, plot_name):
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
        W_conv1 = weight_variable([5, 5, 3, 32])
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

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        offset = 0
        print("Beginning training epochs")
        epoch_start = None
        epoch_end = None
        epoch_time = 0.0
        for i in range(EPOCHS):
            # obtain training batch
            train_image_batch, train_label_batch, offset = get_batch(train, BATCH_SIZE, offset)

            #print("len_batch: %s, x shape: %s, type: %s" % (len(train_image_batch), 
            #                                                train_image_batch[0].shape, 
            #                                                type(train_image_batch[0])))

            feed_dict_train = {x: train_image_batch, y_true: train_label_batch, keep_prob: DROPOUT}
            sess.run(train_op, feed_dict=feed_dict_train)
            
            # report loss, accuracy every few hundred iterations
            if i % 200 == 0:
                if epoch_start is not None:
                    epoch_end = time.time()
                    epoch_time = epoch_end - epoch_start
                else:
                    epoch_start = time.time()
                    epoch_end = time.time()
                feed_dict_mini = {x: train_image_batch, y_true: train_label_batch, keep_prob: 1.0}
                loss, acc = sess.run([loss_op, accuracy], feed_dict=feed_dict_mini)

                print("epoch %d, mini-batch loss %g, training accuracy %g, time %g" % (i, loss, acc, epoch_time))

        # calculate accuracy for test set
        test_image_batch, test_label_batch, _ = get_batch(test, len(test), 0)
        feed_dict_test = {x: test_image_batch, y_true: test_label_batch, keep_prob: 1.0}
        acc = sess.run(accuracy, feed_dict_test)
        print("test accuracy = %g" % (acc))

        # generate confustion matrix
        classification = y_pred_cls.eval(feed_dict_test)
        cm = tf_confusion_matrix(classification, test_label_batch, CLASSES)

        print("Confusion Matrix:\n%s" % (cm))

        plot_title = "%s confusion matrix, e=%s, learn rate=%s" % (model_name, EPOCHS, LEARNING_RATE)
        plot_confusion_matrix(cm, CLASSES, "plots/" + plot_name + ".png", title=plot_title)

        saver = tf.train.Saver()
        save_path = saver.save(sess, "./checkpoints/%s.ckpt" % (model_name))
        print("Model saved in path: %s" % save_path)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def parse_csv_data(data_file):
    '''parses the csv data and returns a dictionary of classes
    data_dict is formatted:
    { "length": number of samples in dictionary,
      "class" : [
                [array(image 1 pixel values), array(labels one hot)],
                [array(image 2 pixel values), array(labels one hot)],
                ...
                ]
    }
    '''
    data_dict = {'length': 0}

    for c in CLASSES:
        data_dict[c] = []

    with open(data_file, 'rb') as f:
        data_reader = csv.reader(f)
        for row in data_reader:
            label_one_hot = np.array(row[0:len(CLASSES)])
            image_data = np.array(row[len(CLASSES):])
            index = np.argmax(label_one_hot)
            class_name = CLASSES[index]
            data_dict[class_name].append([image_data, label_one_hot])

            data_dict['length'] = data_dict['length'] + 1

    return data_dict


def balanced_samples(data, train_size):
    '''returns a list of samples balanced across all classes'''
    class_counts = {}
    train_samples = []
    test_samples = []
    train_indices = {}
    test_indices = {}
    train_sample_counts = {}
    test_sample_counts = {}

    num_samples = data['length']
    num_train_samples = int(num_samples * train_size)
    num_test_samples = num_samples - num_train_samples

    # collect the lengths of each class
    for k in data.keys():
        if k in CLASSES:
            class_counts[k] = len(data[k])
            print("class count for %s = %s" % (k, len(data[k])))

            indices = range(0, len(data[k]))
            train_indices[k] = random.sample(indices, int(len(indices) * train_size))
            test_indices[k] = [ x for x in indices if x not in train_indices ]

    # now collect the samples
    c = 0
    i = 0
    while len(train_samples) < num_train_samples:
        cur_class = CLASSES[c]
        index = train_indices[cur_class][i % len(train_indices[cur_class])]
        train_samples.append(data[cur_class][index])

        if cur_class not in train_sample_counts.keys():
            train_sample_counts[cur_class] = 0
        else:
            train_sample_counts[cur_class] += 1

        c += 1
        i += 1
        if c >= len(CLASSES):
            c = 0

    c = 0
    i = 0
    while len(test_samples) < num_test_samples:
        cur_class = CLASSES[c]
        index = test_indices[cur_class][i % len(test_indices[cur_class])]
        test_samples.append(data[cur_class][index])

        if cur_class not in test_sample_counts.keys():
            test_sample_counts[cur_class] = 0
        else:
            test_sample_counts[cur_class] += 1

        c += 1
        i += 1
        if c >= len(CLASSES):
            c = 0

    print("Train sample makeup - %s samples:" % (num_train_samples))
    for k in CLASSES:
        print("%s - %s" % (k, train_sample_counts[k]))

    print("Test sample makeup - %s samples:" % (num_test_samples))
    for k in CLASSES:
        print("%s - %s" % (k, test_sample_counts[k]))

    return train_samples, test_samples


def main():
    """trains and tests a CNN given a set of tfrecord files"""
    data_dir = '/home/assistive-robotics/object_naming_dataset/jordan-code/data/'  # where to get the data
    num_classes = len(CLASSES)
    data_name = "top_img_raw"
    data_type = img_resize_dtype
    data_file = data_dir + data_name + "_64x64.csv"

    data_dict = parse_csv_data(data_file)

    for t in TRAIN_SET_SIZE:
        # create train and test sets
        train, test = balanced_samples(data_dict, t)

        if SHUFFLE:
            random.shuffle(train)
            random.shuffle(test)
        print("===================================\nt = %s" % (t))
        print("training with %s samples, testing with %s samples" % (len(train), len(test)))

        plot_name = "img_raw_" + str(t)
        train_cnn(train, test, data_name, data_type, plot_name)







if __name__ == "__main__":
    main()
