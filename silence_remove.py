#! /usr/bin/env python
# encoding: utf-8


###########################################################################
# This vad will be applied to low sample rate data: 16kHz and high sample rate data: 192kHz
###########################################################################
import numpy
import scipy.io.wavfile as wf
import sys
import glob
import pathlib



class VoiceActivityDetection:

    def __init__(self):
        self.__step = 400
        self.__buffer_size = 400
        self.__buffer = numpy.array([], dtype=numpy.int16)
        self.__out_buffer = numpy.array([], dtype=numpy.int16)

        self.__hstep = 4800                         # designed for high frequency data
        self.__hbuffer_size = 4800
        self.__hbuffer = numpy.array([], dtype=numpy.int16)
        self.__hout_buffer = numpy.array([], dtype=numpy.int16)

        self.__n = 0
        self.__VADthd = 0.
        self.__VADn = 0.
        self.__silence_counter = 0
        self.__iframe = 0

    # Voice Activity Detection
    # Adaptive threshold
    def vad(self, _frame):
        frame = numpy.array(_frame) ** 2.
        # frame = numpy.array(_frame) 
        result = True
        threshold = 0.25
        thd = numpy.min(frame) + numpy.ptp(frame) * threshold
        self.__VADthd = (self.__VADn * self.__VADthd + thd) / float(self.__VADn + 1.)
        self.__VADn += 1.

        # print("Mean is {}\n thd is {}".format(numpy.mean(frame), self.__VADthd))
        if numpy.mean(frame) <= self.__VADthd:
            self.__silence_counter += 1
        else:
            self.__silence_counter = 0

        if self.__silence_counter > 25:
            result = False
        return result

    # Push new audio samples into the buffer.
    def add_samples(self, data, hdata):
        self.__buffer = numpy.append(self.__buffer, data)
        self.__hbuffer = numpy.append(self.__hbuffer, hdata)
        result = len(self.__buffer) >= self.__buffer_size
        # print('__buffer size %i'%self.__buffer.size)
        return result

    # Pull a portion of the buffer to process
    # (pulled samples are deleted after being
    # processed
    def get_frame(self):
        window = self.__buffer[:self.__buffer_size]
        self.__buffer = self.__buffer[self.__step:]
        self.__iframe = self.__iframe + 1
        # print('__buffer size %i'%self.__buffer.size)
        hwindow = self.__hbuffer[:self.__hbuffer_size]
        self.__hbuffer = self.__hbuffer[self.__hstep:]
        # self.__iframe = self.__iframe + 1
        return window, hwindow

    # Adds new audio samples to the internal
    # buffer and process them
    def process(self, data, hdata):
        if self.add_samples(data, hdata):
            while len(self.__buffer) >= self.__buffer_size:
                # Framing
                window, hwindow = self.get_frame()
                # print('window size %i'%window.size)
                result = self.vad(window)
                if result:
                    self.__out_buffer = numpy.append(self.__out_buffer, window)
                    self.__hout_buffer = numpy.append(self.__hout_buffer, hwindow)
                    # remove high frequency data only based on low frequency results
                # else:
                #     print("current frame is {}".format(self.__iframe))
                # if self.vad(window):  # speech frame
                #     self.__out_buffer = numpy.append(self.__out_buffer, window)
                # print('__out_buffer size %i'%self.__out_buffer.size)

    def get_voice_samples(self):
        return self.__out_buffer, self.__hout_buffer


if __name__ == '__main__':
    low_files = glob.glob("../oakland-dataset/lowfrequency/*/*.wav")
    high_files = glob.glob("../oakland-dataset/dataset_1/*/*.wav")
    print(low_files)
    print(high_files)
    i=0
    for low_file, high_file in zip(low_files, high_files):
        wav = wf.read(low_file)
        hwav = wf.read(high_file)
        sr = wav[0]
        c0 = wav[1]
        hsr = hwav[0]
        hc0 = hwav[1]
        vad = VoiceActivityDetection()
        vad.process(c0, hc0)
        voice_samples, hvoice_samples = vad.get_voice_samples()
        outfile = low_file[:-4] + "cut.wav"
        wf.write(outfile, sr, voice_samples)

        houtfile = high_file[:-4] + "highcut.wav"
        wf.write(houtfile, hsr, hvoice_samples)

        i = i + 1
