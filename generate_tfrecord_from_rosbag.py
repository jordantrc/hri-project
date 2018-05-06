#!/usr/bin/env python

# generate_tfrecord_from_rosbag.py
# Madison Clark-Turner
# 12/11/2017
#
# Updated by Jordan Chadwick
# 4/4/18
#
# Updated for the two-camera dataset


import random
import tensorflow as tf
import numpy as np

# file IO
import heapq
import rosbag
import os
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


class Event:

    def __init__(self, flip, serial=None, label=None, start=None, end=None):
        self.serial = serial
        self.label = label
        self.start = start
        self.end = end
        self.packager = DQNPackager(flip=flip)
        self.msg_count = 0
        self.complete = False

    def __repr__(self):
        string = "EVENT:\n%s\n%s\n%s\n%s\n" % (self.label, self.serial, self.start, self.end)
        return string

    def equals(self, label, serial):
        if self.label == label and self.serial == serial:
            return True
        else:
            return False

    def valid(self):
        if self.serial is not None and self.label is not None and self.start is not None and self.end is not None:
            return True
        else:
            return False


def read_timing_file(filename, flip):
    '''reads timing file
    format of the file is:
    label_sequence_[s|e] time
    s = start
    e = end
    time is formatted as ss.mm
    '''
    ifile = open(filename, 'r')
    events = []
    line = ifile.readline()
    while len(line) != 0:
        line = line.split()
        try:
            event_data = line[0]
            event_time = float(line[1])
            event_time = rospy.Duration(event_time)

            # get the label from the event data
            if event_data.count('_') == 2:
                event_label, event_serial, event_type = event_data.split('_')
            else:
                event_label, event_type = event_data.split('_')
                event_serial = 0

            if event_type == 'e':
                for e in events:
                    if e.equals(event_label, event_serial):
                        e.end = event_time
            else:
                event = Event(flip, event_serial, event_label, event_time, None)
                events.append(event)
        except IndexError:
            print("## ERROR ## timing file [%s] line [%s] invalid" % (filename, str(line)))
        line = ifile.readline()
    # check all events
    for e in events:
        if not e.valid():
            print("Invalid event: %s" % (e))
            print("Input file = %s" % (filename))
    print("====DEBUG=====\nevents = [%s]\n====/DEBUG=====" % (events))
    ifile.close()
    return events


# Format the data to be read
def processData(inp, data_type):
    #print(tf.shape(inp))
    data_s = tf.reshape(inp, [-1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]])
    return tf.cast(data_s, tf.uint8)


# Use for visualizing Data Types
# taken from: https://github.com/AssistiveRoboticsUNH/TR-LfD/blob/master/
# itbn_lfd/itbn_model/src/itbn_classifier/tools/generate_itbn_tfrecords.py
def show(data, d_type):
    tout = []
    out = []
    # print("===DEBUG===\nshow fn\ndata.shape=%s\n===/DEBUG===" % (data))
    frame_limit = 0
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
    if frame_limit != 0 and data.shape[0] % frame_limit != 0:
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

    return None


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

    generated_files = []

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
    #      READ FILE      #
    #######################
    # adapted from:
    # https://github.com/AssistiveRoboticsUNH/TR-LfD/blob/master/itbn_lfd/itbn_model/src/itbn_classifier/tools/generate_itbn_tfrecords.py
    events = read_timing_file(timing_filename, flip)

    start_time = None
    serial = 0

    for topic, msg, t in bag.read_messages(topics=topic_names):
        # TODO
        # Need to fix this section, use the per-event packager

        # DEBUG
        # print("===DEBUG===\ntopic = %s\nmsg = %s\nt = %s\n===/DEBUG===" % (topic, msg, t))
        if start_time is None:
            start_time = t

        for e in events:
            if t >= e.start + start_time and t < e.end + start_time:
                # accumulate frames
                if(topic == topic_names[1]):
                    e.packager.topImgCallback(msg)
                elif(topic == topic_names[2]):
                    e.packager.botImgCallback(msg)
                elif(topic == topic_names[3]):
                    e.packager.audCallback(msg)
                e.msg_count += 1

            elif t >= e.end + start_time and not e.complete:
                # prepare the sample, only if messages added
                if e.msg_count > 0:
                    e.packager.formatOutput()
                    ex = make_sequence_example(
                        e.packager.getTopImgStack(), img_dtype,
                        e.packager.getTopGrsStack(), grs_dtype,
                        e.packager.getBotImgStack(), img_dtype,
                        e.packager.getBotGrsStack(), grs_dtype,
                        e.packager.getTopPntStack(), pnt_dtype,
                        e.packager.getBotPntStack(), pnt_dtype,
                        e.packager.getAudStack(), aud_dtype,
                        e.label)

                    # write TFRecord data to file
                    output_path = "%s%s_%s_%s.tfrecord" % (out_dir, out_filename, serial, e.label)
                    writer = tf.python_io.TFRecordWriter(output_path)
                    writer.write(ex.SerializeToString())
                    writer.close()
                    print("WROTE %s frames from %s to %s" % (e.msg_count, bag_filename, output_path))
                    print("%s" % (e))
                    generated_files.append(output_path)
                    e.complete = True
                    e.packager.reset()

                    serial += 1

    bag.close()
    return generated_files


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
    gen_single_file = False
    view_single_file = False
    process_all_files = False
    check_files = True

    rospy.init_node('gen_tfrecord', anonymous=True)

#############################

    # USAGE: generate a single file and store it as a scrap.tfrecord; Used for Debugging

    root_dir = "/home/assistive-robotics/object_naming_dataset/"
    generated_files = []
    tfrecord_output_dir = root_dir + "tfrecords/"
    random.seed(0)

    if gen_single_file:
        print("GENERATING A SINGLE TEST FILE...")
        bagfile = root_dir + "bags/subject1/successA/sa_0.bag"
        timefile = root_dir + "temp_info/subject1/successA/sa_0.txt"

        # redefine tfrecords output
        tfrecord_output_dir = root_dir + "tfrecords-test/"
        generated_files = gen_TFRecord_from_file(out_dir=tfrecord_output_dir, out_filename="sa_0", bag_filename=bagfile,
                                                 timing_filename=timefile, flip=False)
    elif process_all_files:
        # look for subjects in the bags directory
        bag_dir = root_dir + "bags/"
        temp_dir = root_dir + "temp_info/"
        subjects = os.listdir(bag_dir)

        for su in subjects:
            subject_bag_dir = bag_dir + su + "/"
            subject_temp_dir = temp_dir + su + "/"

            # process all samples for the subject
            samples = os.listdir(subject_bag_dir)
            for sa in samples:
                sample_bag_dir = subject_bag_dir + sa + "/"
                sample_temp_dir = subject_temp_dir + sa + "/"

                # process all segments for the sample
                segments = os.listdir(sample_bag_dir)
                for se in segments:
                    flip_segment = False
                    if random.randint(0, 1) == 1:
                        # flip with 50/50 chance
                        flip_segment = True

                    base_name, extension = se.split(".")

                    segment_bag_path = sample_bag_dir + se
                    segment_temp_path = sample_temp_dir + base_name + ".txt"

                    print("GENERATING TFRecords for %s..." % (segment_bag_path))
                    if flip_segment:
                        outfile_name = "%s_%s_%s_flip" % (su, sa, base_name)
                    else:
                        outfile_name = "%s_%s_%s" % (su, sa, base_name)
                    new_files = gen_TFRecord_from_file(out_dir=tfrecord_output_dir, out_filename=outfile_name,
                                           bag_filename=segment_bag_path, timing_filename=segment_temp_path,
                                           flip=flip_segment)
                    generated_files.extend(new_files)

#############################

    # USAGE: read contents of scrap.tfrecord; Used for Debugging
    file_set = set()
    if view_single_file:
        file_set.add(generated_files[0])
    elif check_files:
        # view a randomly picked sample, 5% of the files
        # backup method in case this is a re-run
        if len(generated_files) == 0:
            generated_files = os.listdir(tfrecord_output_dir)
        assert len(generated_files) > 0 
        num_files = int(len(generated_files) * 0.05)
        random_sample = [ generated_files[i] for i in sorted(random.sample(xrange(len(generated_files)), num_files)) ]
        for r in random_sample:
            file_set.add(tfrecord_output_dir + r)
        print("CHECKING %s randomly sampled files" % (num_files))

    record = 0
    for f in file_set:
        print("READING %s" % (f))
        # todo fix this need the filename
        filename = os.path.split
        coord = tf.train.Coordinator()
        filename_queue = tf.train.string_input_producer([f])

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            # parse TFrecord
            context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
            threads = tf.train.start_queue_runners(coord=coord)

            seq_len = context_parsed["length"]  # sequence length
            label = context_parsed["label"]  # class labels
            label = tf.one_hot(label, depth=len(CLASSES))
            print("===DEBUG===\nseq_len = %s\nlabel = %s\n===/DEBUG===" % (seq_len, label))

            top_img_raw = processData(sequence_parsed["top_img_raw"], img_dtype)
            top_grs_raw = processData(sequence_parsed["top_grs_raw"], grs_dtype)
            bot_img_raw = processData(sequence_parsed["bot_img_raw"], img_dtype)
            bot_grs_raw = processData(sequence_parsed["bot_grs_raw"], grs_dtype)
            top_opt_raw = processData(sequence_parsed["top_opt_raw"], pnt_dtype)
            bot_opt_raw = processData(sequence_parsed["bot_opt_raw"], pnt_dtype)
            #aud_raw = processData(sequence_parsed["aud_raw"], aud_dtype)

            # set range to value > 1 if multiple TFRecords stored in a single file
            l, ti, tg, bi, bg, to, bo, la = sess.run(
                [seq_len,
                 top_img_raw,
                 top_grs_raw,
                 bot_img_raw,
                 bot_grs_raw,
                 top_opt_raw,
                 bot_opt_raw,
                 label
                 ])
            print("SHAPES: ", l, ti.shape, tg.shape, bi.shape, bg.shape, to.shape, bo.shape, la.shape)

            coord.request_stop()
            coord.join(threads)

            # display the contents of the optical flow and image files
            show_from = 0
            top_opt = show(to[show_from:], pnt_dtype)
            bot_opt = show(bo[show_from:], pnt_dtype)
            top_img = show(ti[show_from:], img_dtype)
            bot_img = show(bi[show_from:], img_dtype)
            top_grs_img = show(tg[show_from:], grs_dtype)
            bot_grs_img = show(bg[show_from:], grs_dtype)
            #audio_img = show(a[show_from:], aud_dtype)

            cv2.imwrite("./images/" + f + "top_opt" + str(record) + ".jpg", top_opt)
            cv2.imwrite("./images/bot_opt" + str(record) + ".jpg", bot_opt)
            cv2.imwrite("./images/top_img" + str(record) + ".jpg", top_img)
            cv2.imwrite("./images/bot_img" + str(record) + ".jpg", bot_img)
            cv2.imwrite("./images/top_grs_img" + str(record) + ".jpg", top_grs_img)
            cv2.imwrite("./images/bot_grs_img" + str(record) + ".jpg", bot_grs_img)
            #cv2.imwrite("./images/audio_img" + str(record) + ".jpg", audio_img)
            record += 1
