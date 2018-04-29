#!/bin/env/python
#
# Library for the Object Naming dataset CNNs
#

import tensorflow as tf
import os

from basic_tfrecord_rw import *


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
        self.dataset = tf.data.TFRecordDataset(filenames)



