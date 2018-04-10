#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from constants import *


def make_sequence_example(top_img_input, top_img_data, bot_img_input, bot_img_data, top_opt_input,
                          top_opt_data, bot_opt_input, bot_opt_data, aud_input, aud_data, timing_dict,
                          example_id):
    '''generates the tfrecord sequence
    from: https://github.com/AssistiveRoboticsUNH/TR-LfD/blob/master/itbn_lfd/itbn_model/src/itbn_classifier/common/itbn_tfrecord_rw.py
    '''
    ex = tf.train.SequenceExample()
    # non-sequential features
    sequence_length = top_opt_input.shape[1]

    ex.context.feature['length'].int64_list.value.append(sequence_length)

    ex.context.feature["top_img_h"].int64_list.value.append(top_img_data["cmp_h"])
    ex.context.feature["top_img_w"].int64_list.value.append(top_img_data["cmp_w"])
    ex.context.feature["top_img_c"].int64_list.value.append(top_img_data["num_c"])

    ex.context.feature["bot_img_h"].int64_list.value.append(bot_img_data["cmp_h"])
    ex.context.feature["bot_img_w"].int64_list.value.append(bot_img_data["cmp_w"])
    ex.context.feature["bot_img_c"].int64_list.value.append(bot_img_data["num_c"])

    ex.context.feature["top_pnt_h"].int64_list.value.append(top_opt_data["cmp_h"])
    ex.context.feature["top_pnt_w"].int64_list.value.append(top_opt_data["cmp_w"])
    ex.context.feature["top_pnt_c"].int64_list.value.append(top_opt_data["num_c"])

    ex.context.feature["bot_pnt_h"].int64_list.value.append(bot_opt_data["cmp_h"])
    ex.context.feature["bot_pnt_w"].int64_list.value.append(bot_opt_data["cmp_w"])
    ex.context.feature["bot_pnt_c"].int64_list.value.append(bot_opt_data["num_c"])

    ex.context.feature["aud_h"].int64_list.value.append(aud_data["cmp_h"])
    ex.context.feature["aud_w"].int64_list.value.append(aud_data["cmp_w"])
    ex.context.feature["aud_c"].int64_list.value.append(aud_data["num_c"])

    ex.context.feature["example_id"].bytes_list.value.append(example_id)

    timing_labels, timing_values = "", []
    for k in timing_dict.keys():
        # print("===DEBUG===\ntiming_dict[%s] = %s\n===/DEBUG===" % (k, timing_dict[k]))
        timing_labels += k + "/"
        timing_values.append(timing_dict[k])

    ex.context.feature["timing_labels"].bytes_list.value.append(timing_labels)

    # Feature lists for input data
    def load_array(example, name, data, dtype):
        fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
        fl_data.append(np.asarray(data).astype(dtype).tostring())

    load_array(ex, "top_img_raw", top_img_input, np.uint8)
    load_array(ex, "bot_img_raw", bot_img_input, np.uint8)
    load_array(ex, "top_opt_raw", top_opt_input, np.uint8)
    load_array(ex, "bot_opt_raw", bot_opt_input, np.uint8)
    load_array(ex, "aud_raw", aud_input, np.uint8)
    load_array(ex, "timing_values", timing_values, np.int16)

    return ex


# READ
def parse_sequence_example(filename_queue):
    # reads a TFRecord into its constituent parts
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64),

        "top_img_h": tf.FixedLenFeature([], dtype=tf.int64),
        "top_img_w": tf.FixedLenFeature([], dtype=tf.int64),
        "top_img_c": tf.FixedLenFeature([], dtype=tf.int64),

        "bot_img_h": tf.FixedLenFeature([], dtype=tf.int64),
        "bot_img_w": tf.FixedLenFeature([], dtype=tf.int64),
        "bot_img_c": tf.FixedLenFeature([], dtype=tf.int64),

        "top_pnt_h": tf.FixedLenFeature([], dtype=tf.int64),
        "top_pnt_w": tf.FixedLenFeature([], dtype=tf.int64),
        "top_pnt_c": tf.FixedLenFeature([], dtype=tf.int64),

        "bot_pnt_h": tf.FixedLenFeature([], dtype=tf.int64),
        "bot_pnt_w": tf.FixedLenFeature([], dtype=tf.int64),
        "bot_pnt_c": tf.FixedLenFeature([], dtype=tf.int64),

        "aud_h": tf.FixedLenFeature([], dtype=tf.int64),
        "aud_w": tf.FixedLenFeature([], dtype=tf.int64),
        "aud_c": tf.FixedLenFeature([], dtype=tf.int64),

        "timing_labels": tf.FixedLenFeature([], dtype=tf.string),

        "example_id": tf.FixedLenFeature([], dtype=tf.string)
    }

    sequence_features = {
        "top_img_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "bot_img_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "top_opt_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "bot_opt_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "aud_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "timing_values": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    sequence_data = {
        "top_img_raw": tf.decode_raw(sequence_parsed["top_img_raw"], tf.uint8),
        "bot_img_raw": tf.decode_raw(sequence_parsed["bot_img_raw"], tf.uint8),
        "top_opt_raw": tf.decode_raw(sequence_parsed["top_opt_raw"], tf.uint8),
        "bot_opt_raw": tf.decode_raw(sequence_parsed["bot_opt_raw"], tf.uint8),
        "aud_raw": tf.decode_raw(sequence_parsed["aud_raw"], tf.float64),
        "timing_values": tf.decode_raw(sequence_parsed["timing_values"], tf.int16),
    }

    return context_parsed, sequence_data


def set_input_shape(arr, data_type):
    return np.reshape(arr, (BATCH_SIZE, -1, data_type["size"] * data_type["size"] * data_type["num_c"]))
