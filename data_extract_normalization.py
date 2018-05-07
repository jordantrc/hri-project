#!
#
# Extracts and normalizes data from grayscale tfrecords
#
# Saves data in CSV format
#

import os
import csv
import cv2 as cv
import tensorflow as tf
import time

from object_naming_grs_cnn import SampleGenerator
from constants import *
from basic_tfrecord_rw import *


IMAGE_STACK_SIZE = 10


# main
tfrecord_dir = '/home/assistive-robotics/object_naming_dataset/tfrecords/'
tfrecords = os.listdir(tfrecord_dir)

tfrecords = [ tfrecord_dir + x for x in tfrecords ]

#opt = []
grs = []

for i, t in enumerate(tfrecords):

    start = time.time()
    #opt.append(SampleGenerator.get_sample_from_tfrecord(t, 'top_opt_raw', pnt_dtype))
    grs.append(SampleGenerator.get_sample_from_tfrecord(t, 'top_grs_raw', grs_dtype))
    end = time.time()
    print("%s/%s Extracted data from %s in %s seconds" % (i, len(tfrecords), t, end - start))


#print(opt[0])
print(grs[0])

#with open('data/top_opt_raw.csv', 'wb') as csvfile:
#    writer = csv.writer(csvfile)
#    for r in opt:
#        img_data = r[0].tolist()
#        label_one_hot = r[1].tolist()
#        label_one_hot.extend(img_data)
#        writer.writerow(label_one_hot)

with open('data/top_grs_raw.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for r in grs:
        img_data = r[0].tolist()
        label_one_hot = r[1].tolist()
        label_one_hot.extend(img_data)
        writer.writerow(label_one_hot)
