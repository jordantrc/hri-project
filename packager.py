# dqn_packager_fast.py
# Madison Clark-Turner
# 10/14/2017
#
# Modified by Jordan Chadwick
# Pre-processing for the two-camera dataset.
#

import tensorflow as tf
import numpy as np

from constants import *

import math
import threading

# ROS
import rospy, rospkg
from std_msgs.msg import Int8
from sensor_msgs.msg import Image

# image pre-processing and optical flow generation
import cv2
from cv_bridge import CvBridge, CvBridgeError

# audio pre-processing
from nao_msgs.msg import AudioBuffer
from noise_subtraction import reduce_noise
from scipy import signal
import librosa, librosa.display

# DQN
from dqn_model import DQNModel

topic_names = [
    '/action_started',
    '/nao_robot/camera/top/camera/image_raw',
    '/nao_robot/camera/bottom/camera/image_raw',
    '/nao_robot/microphone/naoqi_microphone/audio_raw'
]

'''
DQN Packager listens to the topics for images and audio.
Processes those inputs into sequences and passes the result to
the DQN model.
'''


class DQNPackager:

    def __init__(self, dqn=None, flip=False):
        # dqn model
        self.__dqn = dqn
        self.__flip = flip

        # variables for tracking images received
        self.__lock = threading.Lock()
        self.reset()

        # variables for optical flow
        self.frame1 = []
        self.previous_frame, self.hsv = None, None

        # variables for audio
        self.counter = 0
        # src_dir = rospkg.RosPack().get_path('deep_reinforcement_abstract_lfd') + '/src/dqn/'
        src_dir = './'
        self.__face_cascade = cv2.CascadeClassifier(src_dir + 'haarcascade_frontalface_default.xml')
        self.rate = 16000  # Sampling rate

        # subscribers
        QUEUE_SIZE = 1
        self.sub_act = rospy.Subscriber(topic_names[0],
                                        Int8, self.actCallback, queue_size=QUEUE_SIZE)

        self.sub_top_img = rospy.Subscriber(topic_names[1],
                                            Image, self.topImgCallback, queue_size=QUEUE_SIZE)

        self.sub_bot_img = rospy.Subscriber(topic_names[2],
                                            Image, self.botImgCallback, queue_size=QUEUE_SIZE)

        self.sub_aud = rospy.Subscriber(topic_names[3],
                                        AudioBuffer, self.audCallback, queue_size=QUEUE_SIZE)

    def getRecentAct(self):
        return self.__most_recent_act

    def getTopImgStack(self):
        return self.__topImgStack

    def getTopGrsStack(self):
        return self.__topGrsStack

    def getBotImgStack(self):
        return self.__botImgStack

    def getBotGrsStack(self):
        return self.__botGrsStack

    def getTopPntStack(self):
        return self.__topPntStack

    def getBotPntStack(self):
        return self.__botPntStack

    def getAudStack(self):
        return self.__audStack

    def getFrameCount(self):
        if type(self.__topImgStack) == int:
            return 0
        return len(self.__topImgStack)

    ############################
    # Collect Data into Frames #
    ############################

    def clearMsgs(self):
        self.__recent_msgs = [False] * 3

    def reset(self, already_locked=False):
        if not already_locked:
            self.__lock.acquire()
        self.clearMsgs()
        self.__topImgStack = 0
        self.__topGrsStack = 0
        self.__botImgStack = 0
        self.__botGrsStack = 0
        self.__topPntStack = 0
        self.__botPntStack = 0
        self.__audStack = 0

        self.frame1 = []
        self.previous_frame, self.hsv = None, None

        if not already_locked:
            self.__lock.release()

    def actCallback(self, msg):
        self.__most_recent_act = msg
        self.checkMsgs()
        return

    def topImgCallback(self, msg):
        self.__recent_msgs[0] = msg
        self.checkMsgs()
        return

    def botImgCallback(self, msg):
        self.__recent_msgs[1] = msg
        self.checkMsgs()
        return

    def audCallback(self, msg):
        self.__recent_msgs[2] = msg
        self.checkMsgs()
        return

    def formatAudMsg(self, aud_msg):
        # shapes the audio file for use later
        data = aud_msg.data
        data = np.reshape(data, (-1, 4))
        data = data.transpose([1, 0])

        return data[0]

    def checkMsgs(self):
        # may need to use mutexes on self.__recent_msgs
        self.__lock.acquire()
        if False in self.__recent_msgs:
            self.__lock.release()
            return

        # organize and send data
        # print("__recent_msgs length = %s" % (len(self.__recent_msgs)))
        top_img = self.__recent_msgs[0]
        bot_img = self.__recent_msgs[1]
        aud = self.formatAudMsg(self.__recent_msgs[2])  # process all audio data together prior to sending

        if(type(self.__topImgStack) == int):
            self.__topImgStack = [top_img]
            self.__botImgStack = [bot_img]
            self.__audStack = [aud]
        else:
            self.__topImgStack.append(top_img)
            self.__botImgStack.append(bot_img)
            self.__audStack.append(aud)

        self.clearMsgs()
        self.__lock.release()

    ###############
    # Format Data #
    ###############

    def formatImgBatch(self, img_stack, name=""):
        # pre-process the RGB input and generate the optical flow
        img_out, grs_out, pnt_out = [], [], []

        for i, x in enumerate(img_stack):
            # img is RGB, grs is grayscale
            img, grs = self.formatImg(x)
            if i == 0:
                self.frame1 = img
            img_out.append(np.asarray(img).flatten())
            grs_out.append(np.asarray(grs).flatten())
            pnt_out.append(self.formatOpt(img))

        # reset self.previous_frame
        self.previous_frame = None

        return img_out, grs_out, pnt_out

    def formatImg(self, img_msg):
        # pre-process the image data to crop it to an appropriate size

        # convert image to cv2 image
        img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")

        # identify location of face if possible using Haarcascade and
        # crop to center on it.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.__face_cascade.detectMultiScale(gray, 1.3, 5)
        x, y, w, h = -1, -1, -1, -1

        # if face is locataed to the edge set then crop from the opposite side
        buff = img.shape[1] - img.shape[0]

        if(len(faces) > 0):
            for (xf, yf, wf, hf) in faces:
                if wf * hf > w * h:
                    x, y, w, h = xf, yf, wf, hf

            if(x >= 0 and y >= 0 and x + w < 320 and y + h < 240):
                y, h = 0, img.shape[0]
                mid = x + (w / 2)
                x, w = mid - (img.shape[0] / 2), img.shape[0]
                if(x < 0):
                    x = 0
                elif(x > buff):
                    x = buff / 2
                img = img[y: y + h, x: x + w]
        else:
            # if no face visible set crop image to center of the video
            diff = img.shape[1] - img.shape[0]
            img = img[0:img.shape[0], (diff / 2):(img.shape[1] - diff / 2)]

        # resize image to 299 x 299
        y_mod = 1 / (img.shape[0] / float(img_dtype["cmp_h"]))
        x_mod = 1 / (img.shape[1] / float(img_dtype["cmp_w"]))
        img = cv2.resize(img, None, fx=x_mod, fy=y_mod, interpolation=cv2.INTER_CUBIC)

        if(self.__flip):
            # if flip set to true then mirror the image horizontally
            img = np.flip(img, 1)

        gray_final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img, gray_final

    def formatOpt(self, img_src):
        # generate optical flow
        mod = pnt_dtype["cmp_h"] / float(img_dtype["cmp_h"])
        img = cv2.resize(img_src.copy(), None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)
        
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        opt_img = np.zeros(img.shape)[..., 0]

        # generate optical flow
        frame2 = img
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        next_frame_size = "%shx%sw" % (next_frame.shape[0], next_frame.shape[1])
        prev_frame_size = "%shx%sw" % (self.previous_frame.shape[0], self.previous_frame.shape[1])

        # print("===DEBUG===\nprev_frame_size = %s\nnext_frame_size = %s\n===/DEBUG===" % (next_frame_size, prev_frame_size))
        flow = cv2.calcOpticalFlowFarneback(self.previous_frame, next_frame, None, 0.5, 1, 2, 5, 7, 1.5, 1)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # normalize the magnitude to between 0 and 255 (replace with other normalize to prevent precission issues)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        opt_img = mag
        self.previous_frame = next_frame

        return np.asarray(opt_img).flatten()

    def formatAudBatch(self, aud_msg_array, name=""):
        # perform pre-processing on the audio input

        num_frames = len(aud_msg_array)
        # print('===formatAudBatch DEBUG===\nnum_frames = %s\n===/DEBUG===' % (num_frames))
        input_data = np.reshape(aud_msg_array, (num_frames * len(aud_msg_array[0])))
        core_data = input_data

        # mute the first 2 seconds of audio (where the NAO speaks)
        mute_time = 2
        input_data[:np.argmax(input_data) + int(16000 * mute_time)] = 0

        # get the indicies for the noise sample
        noise_sample_s, noise_sample_e = 16000 * (-1.5), -1

        # perform spectral subtraction to reduce noise
        noise = core_data[int(noise_sample_s): noise_sample_e]
        filtered_input = reduce_noise(np.array(core_data), noise)

        # smooth signal
        b, a = signal.butter(3, 0.05)
        filtered_input = signal.lfilter(b, a, filtered_input)
        noise = filtered_input[int(noise_sample_s): noise_sample_e]

        # additional spectral subtraction to remove remaining noise
        filtered_input = reduce_noise(filtered_input, noise)

        # generate spectrogram
        S = librosa.feature.melspectrogram(y=filtered_input, sr=self.rate, n_mels=128, fmax=8000)
        S = librosa.power_to_db(S, ref=np.max)

        # split the spectrogram into A_i. This generates an overlap between
        # frames with as set stride
        stride = S.shape[1] / float(num_frames)
        frame_len = aud_dtype["cmp_w"]

        # pad the entire spectrogram so that overlaps at either end do not fall out of bounds
        min_val = np.nanmin(S)
        empty = np.zeros((S.shape[0], 3))
        empty.fill(min_val)
        empty_end = np.zeros((S.shape[0], 8))
        empty_end.fill(min_val)
        S = np.concatenate((empty, S, empty_end), axis=1)

        split_data = np.zeros(shape=(num_frames, S.shape[0], frame_len), dtype=S.dtype)
        for i in range(0, num_frames):
            split_data[i] = S[:, int(math.floor(i * stride)):int(math.floor(i * stride)) + frame_len]

        # normalize the output to be between 0 and 255
        split_data -= split_data.min()
        split_data /= split_data.max() / 255.0

        return np.reshape(split_data, (num_frames, -1))

    #############
    # Send Data #
    #############

    def formatOutput(self, name=""):
        # Execute pre-processing on all strored input
        top_img_stack, top_grs_stack, top_opt_stack = self.formatImgBatch(self.__topImgStack, name)
        bot_img_stack, bot_grs_stack, bot_opt_stack = self.formatImgBatch(self.__botImgStack, name)

        self.__topImgStack = np.expand_dims(top_img_stack, axis=0)
        self.__topGrsStack = np.expand_dims(top_grs_stack, axis=0)
        self.__topPntStack = np.expand_dims(top_opt_stack, axis=0)
        self.__botImgStack = np.expand_dims(bot_img_stack, axis=0)
        self.__botGrsStack = np.expand_dims(bot_grs_stack, axis=0)
        self.__botPntStack = np.expand_dims(bot_opt_stack, axis=0)
        self.__audStack = np.expand_dims(self.formatAudBatch(self.__audStack, name), axis=0)

    def getNextAction(self, num_prompt, verbose=False):
        # Execute Pre-processing steps and pass data to the DQN
        if self.__dqn is None:
            print("model not provided!")
            return -1

        self.__lock.acquire()
        num_frames = len(self.__imgStack)

        self.formatOutput()

        if(verbose):
            print("Prediction has " + str(num_frames) + " frames")
            print("imgStack has shape: ", self.__imgStack.shape)
            print("pntStack has shape: ", self.__pntStack.shape)
            print("audStack has shape: ", self.__audStack.shape)
            print("num_prompt is " + str(num_prompt))

        # generate the DQN prediction
        nextact = self.__dqn.genPrediction(num_frames, self.__imgStack, self.__pntStack, self.__audStack, num_prompt) + 1

        # clear the input
        self.reset(already_locked=True)

        self.__lock.release()

        if verbose:
            print("nextact: ", nextact)

        return nextact


if __name__ == '__main__':
    packager = DQNPackager()
