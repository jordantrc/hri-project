#!/usr/bin/env python

# generate_tfrecord_from_rosbag.py
# Madison Clark-Turner
# 12/11/2017
#
# Updated by Jordan Chadwick
# 4/4/18
#
# Updated for the two-camera dataset


import tensorflow as tf
import numpy as np

# file IO
import heapq
import rosbag
import os
from os.path import isfile, join
from basic_tfrecord_rw import *

# contains data type information
from constants import *

# used for performing pre-processing steps on rostopics
from packager import *

# rostopic names
topic_names = [
    '/action_started',
    '/nao_robot/camera/top/camera/image_raw',
    '/nao_robot/camera/bottom/camera/image_raw',
    '/nao_robot/microphone/naoqi_microphone/audio_raw'
]

'''
Read contents of a Rosbag and store:
s  - the current observation
a  - action that followed s
s' - the subsequent observation
a' - the action that followed s'
p  - how many prompts had been delivered before s
'''


def read_timing_file(filename):
    '''reads timing file'''
    ifile = open(filename, 'r')
    timing_queue = []
    line = ifile.readline()
    while len(line) != 0:
        line = line.split()
        event_time = float(line[1])
        event_time = rospy.Duration(event_time)
        timing_queue.append((event_time, line[0]))
        line = ifile.readline()
    heapq.heapify(timing_queue)
    ifile.close()
    return timing_queue


# Format the data to be read
def processData(inp, data_type):
    data_s = tf.reshape(inp, [-1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]])
    return tf.cast(data_s, tf.uint8)


# Use for visualizing Data Types
# taken from: https://github.com/AssistiveRoboticsUNH/TR-LfD/blob/master/itbn_lfd/itbn_model/src/itbn_classifier/tools/generate_itbn_tfrecords.py
def show(data, d_type):
    tout = []
    out = []
    for i in range(data.shape[0]):
        imf = np.reshape(data[i], (d_type["cmp_h"], d_type["cmp_w"], d_type["num_c"]))

        limit_size = d_type["cmp_w"]
        frame_limit = 12
        if d_type["name"] == "aud":
            frame_limit = 120

        if (d_type["cmp_w"] > limit_size):
            mod = limit_size / float(d_type["cmp_h"])
            imf = cv2.resize(imf, None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)

        if (imf.shape[2] == 2):
            imf = np.concatenate((imf, np.zeros((d_type["cmp_h"], d_type["cmp_w"], 1))),
                                 axis=2)
            imf[..., 0] = imf[..., 1]
            imf[..., 2] = imf[..., 1]
            imf = imf.astype(np.uint8)

        if (i % frame_limit == 0 and i != 0):
            if (len(tout) == 0):
                tout = out.copy()
            else:
                tout = np.concatenate((tout, out), axis=0)
            out = []
        if (len(out) == 0):
            out = imf
        else:
            out = np.concatenate((out, imf), axis=1)
    if (data.shape[0] % frame_limit != 0):
        fill = np.zeros((d_type["cmp_h"], d_type["cmp_w"] * (frame_limit -
                                                             (data.shape[0] % frame_limit)),
                         d_type["num_c"]))  # .fill(255)
        fill.fill(0)
        out = np.concatenate((out, fill), axis=1)
    if (len(out) != 0):
        if (len(tout) == 0):
            tout = out.copy()
        else:
            tout = np.concatenate((tout, out), axis=0)
        return tout


def gen_TFRecord_from_file(out_dir, out_filename, bag_filename, timing_filename, flip=False):
    # outdir, bagfile, state, name, flip=False, index=-1):
    '''
    out_dir - the desierd output directory
    out_filename - the filename of the generated TFrecord (should NOT include
        suffix)
    bag_filename - the rosbag being read
    flip - whether the img and optical flow data should be flipped horizontally
        or not
    '''

    # packager subscribes to rostopics and does pre-processing
    packager = DQNPackager(flip=flip)

    # read RosBag
    bag = rosbag.Bag(bag_filename)

    # print some debugging info about the bag
    types, topics = bag.get_type_and_topic_info()
    print("===DEBUG===\nBag info:\nnum messages = %s\nstart_time = %s\nend_time = %s\n" %
          (bag.get_message_count(), bag.get_start_time(), bag.get_end_time()))
    print("Topics:")
    for k in topics.keys():
        print("%s - %s messages" % (k, topics[k][1]))
    print("===/DEBUG===")

    #######################
    #  ALTER TIMING INFO  #
    #######################

    # for topic, msg, t in bag.read_messages(topics=['/action_started']):
    #    print("===DEBUG===\ntopic = %s\nmsg = %s\nt = %s\n===/DEBUG===""" % (topic, msg, t))

    #   if(index >= 0 and index <= 4 and msg.data == 1 and state == 1):
    #    msg.data = 2
    #   if(msg.data == 0):
    #        t = t - rospy.Duration(2.5)
    #    if(msg.data == 1):
    #        t = t - rospy.Duration(2.5)
    #    if(msg.data == 2):
    #        t = t - rospy.Duration(1)
    #    time_log.append(t)
    #    all_actions.append(msg)

    # fail early if the timing info was not properly set
    # assert len(time_log) > 0

    #######################
    #   TIMING FILE       #
    #######################

    timing_queue = read_timing_file(timing_filename)
    current_time = heapq.heappop(timing_queue)
    timing_dict = {}

    #######################
    #      READ FILE      #
    #######################
    # adapted from:
    # https://github.com/AssistiveRoboticsUNH/TR-LfD/blob/master/itbn_lfd/itbn_model/src/itbn_classifier/tools/generate_itbn_tfrecords.py
    start_time = None
    all_timing_frames_found = False
    msg_count = 0

    for topic, msg, t in bag.read_messages(topics=topic_names):
        # DEBUG
        # print("===DEBUG===\ntopic = %s\nmsg = %s\nt = %s\n===/DEBUG===" % (topic, msg, t))
        if start_time is None:
            start_time = t

        if not all_timing_frames_found and t > start_time + current_time[0]:
            # add the frame number and timing label to frame dict
            # assumption - frame number is the same for both top and bottom images (they should be)
            timing_dict[current_time[1]] = packager.getFrameCount()
            if len(timing_queue) > 0:
                current_time = heapq.heappop(timing_queue)
            else:
                all_timing_frames_found = True

        if(topic == topic_names[1]):
            packager.topImgCallback(msg)
        elif(topic == topic_names[2]):
            packager.botImgCallback(msg)
        elif(topic == topic_names[3]):
            packager.audCallback(msg)
        msg_count += 1

    packager.formatOutput()

    ex = make_sequence_example(
        packager.getTopImgStack(), img_dtype,
        packager.getBotImgStack(), img_dtype,
        packager.getTopPntStack(), pnt_dtype,
        packager.getBotPntStack(), pnt_dtype,
        packager.getAudStack(), aud_dtype,
        timing_dict,
        timing_filename)

    # write TFRecord data to file
    end_file = ".tfrecord"
    if flip:
        end_file = "_flip" + end_file

    writer = tf.python_io.TFRecordWriter(out_dir + out_filename + end_file)
    writer.write(ex.SerializeToString())
    writer.close()

    packager.reset()

    bag.close()
    print("WROTE %s frames from %s to %s" % (msg_count, bag_filename, out_dir + out_filename))


def check(filenames):
    coord = tf.train.Coordinator()
    filename_queue = tf.train.string_input_producer(filenames, capacity=32)
    out = []
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
        threads = tf.train.start_queue_runners(coord=coord)
        seq_len = context_parsed["length"]  # sequence length
        n = context_parsed["example_id"]
        for i in range(len(filenames) * 2):
            num_frames, name = sess.run([seq_len, n])
            num_frames = num_frames.tolist()

            print(str(i) + '/' + str(len(filenames) * 2), num_frames, name)

            out.append([num_frames, name])

        coord.request_stop()
        sess.run(filename_queue.close(cancel_pending_enqueues=True))
        coord.join(threads)

    return out


if __name__ == '__main__':
    gen_single_file = True
    view_single_file = True
    process_all_files = False

    rospy.init_node('gen_tfrecord', anonymous=True)

#############################

    # USAGE: generate a single file and store it as a scrap.tfrecord; Used for Debugging

    bagfile = os.environ["HOME"] + "/Documents/samples/success/sa_0.bag"
    timefile = os.environ["HOME"] + "/Documents/samples/success/sa_0.txt"

    outfile = os.environ["HOME"] + "/Documents/samples/tfrecords/sa_0.tfrecord"
    outdir = os.environ["HOME"] + "/Documents/samples/tfrecords/"

    if(gen_single_file):
        print("GENERATING A SINGLE TEST FILE...")
        gen_TFRecord_from_file(out_dir=outdir, out_filename="sa_0", bag_filename=bagfile, timing_filename=timefile,
                               flip=False)

#############################

    # USAGE: read contents of scrap.tfrecord; Used for Debugging
    file_set = set()
    if view_single_file:
        file_set.add(outfile)
    elif view_all_files:
        pass

    for f in file_set:
        print("READING %s..." % (f))
        coord = tf.train.Coordinator()
        filename_queue = tf.train.string_input_producer([f])

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            # parse TFrecord
            context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
            threads = tf.train.start_queue_runners(coord=coord)

            seq_len = context_parsed["length"]  # sequence length
            timing_labels = context_parsed["timing_labels"]  # timing labels
            name = context_parsed["example_id"]  # example_id
            timing_values = sequence_parsed["timing_values"]
            print("===DEBUG===\ntiming_values = %s\n===/DEBUG===" % (timing_values))

            top_img_raw = processData(sequence_parsed["top_img_raw"], img_dtype)
            bot_img_raw = processData(sequence_parsed["bot_img_raw"], img_dtype)
            top_opt_raw = processData(sequence_parsed["top_opt_raw"], pnt_dtype)
            bot_opt_raw = processData(sequence_parsed["bot_opt_raw"], pnt_dtype)
            aud_raw = processData(sequence_parsed["aud_raw"], aud_dtype)

            # set range to value > 1 if multiple TFRecords stored in a single file
            for i in range(1):
                l, ti, bi, to, bo, a, tl, tv, n = sess.run(
                    [seq_len,
                     top_img_raw,
                     bot_img_raw,
                     top_opt_raw,
                     bot_opt_raw,
                     aud_raw,
                     timing_labels,
                     timing_values,
                     name
                     ])
                print(ti.shape, bi.shape, to.shape, bo.shape, a.shape, n)

                coord.request_stop()
                coord.join(threads)

                # display the contents of the optical flow file
                show_from = 110
                top_opt = show(ti[show_from:], img_dtype)
                bot_opt = show(bi[show_from:], img_dtype)
                cv2.imshow("top_opt", top_opt)
                cv2.imshow("bot_opt", bot_opt)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

#############################

    # USAGE: write all rosbag demonstrations to TFRecords

    '''
    We assume that the file structure for the demonstartions is ordered as follows:

        -<demonstration_path>
            -<subject_id>_0
                -compliant
                    -<demonstration_name_0>.bag
                    -<demonstration_name_1>.bag
                    -<demonstration_name_2>.bag
                -noncompliant
                    -<demonstration_name_0>.bag

            -<subject_id>_1
            -<subject_id>_2

        -<tfrecord_output_directory>

    '''

    # setup input directory information
    demonstration_path = os.environ["HOME"] + '/' + "Documents/AssistiveRobotics/AutismAssistant/pomdpData/"
    subject_id = "long_sess"

    # setup output directory information
    tfrecord_output_directory = os.environ["HOME"] + '/' + "catkin_ws/src/deep_q_network/tfrecords/long/"

    if(process_all_files):

        for i in range(1, 12):  # each unique subject
            subject_dir = demonstration_path + subject_id + '_'

            if(i < 10):  # fix naming issues with leading 0s
                subject_dir += '0'
            subject_dir += str(i) + '/'

            for s in ["compliant", "noncompliant"]:  # each unique state
                subject_dir += s + '/'

                # get list of demonstration file names
                filename_list = [subject_dir + f for f in os.listdir(subject_dir) if isfile(join(subject_dir, f))]
                filename_list.sort()

                for f in filename_list:
                    # get demonstration name for output file name
                    tag = f
                    while(tag.find("/") >= 0):
                        tag = tag[tag.find("/") + 1:]
                    tag = tag[:-(len(".bag"))]
                    new_name = subject_id + '_' + str(i) + '_' + tag

                    # print files to make it clear process still running
                    print(tag + "......." + new_name)

                    gen_TFRecord_from_file(out_dir=tfrecord_output_directory,
                                           out_filename=new_name, bag_filename=f, flip=False)

                    gen_TFRecord_from_file(out_dir=tfrecord_output_directory,
                                           out_filename=new_name, bag_filename=f, flip=True)
