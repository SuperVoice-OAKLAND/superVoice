import numpy as np
from utils import *
import glob
from data_io_apply8to16 import *


# First function stft:
# extract one fricative consonant bin from one .wav file

# Second function stft2:
# Extract whole stft feature map from given frequency range.

# include all high-frequency files
# run this before running JointTrain.py
# To run this, do :
# python extractHigh.py --cfg=cfg/SincNet_combine_apply8-16.cfg

def stft(file, lower_bound, upper_bound):
    # this function finds the one fricative bin
    # with frequency range from 60 to 800, corresponding to (fs/2)/1025*(80-600) = 7.4kHz - 55.8kHz
    """
    Parameters
    ----------
    file: read file
    lower_bound: lower frequency bound
    upper_bound: higher frequency bound

    Returns
    -------
    one bin of spectrum range from lower bound to higher bound,
    size is : (upper_bound-lower_bound, 1)

    """
    sr, samples = readfile(file)
    Xdb = get_stft(sr=sr, samples=samples)
    max_idx = get_fri_indics(Xdb, lower_bound, upper_bound)
    return Xdb[lower_bound:upper_bound, max_idx]

    # mel_basis = gen_mel(16000, 2048, 10)
    # mfcc = get_mfcc(mel_basis, max_idx, Xdb, 16000, 10, log=True)
    # print(max_idx, mfcc.shape)


def stft2(file, lower_bound, upper_bound):
    # this function get the whole stft result with given frequency range.
    """
    Parameters
    ----------
    file: read file
    lower_bound: lower frequency as Hz
    upper_bound: higher frequency as Hz

    Returns
    -------
    one bin of spectrum range from lower bound to higher bound,
    size is : (upper_bound-lower_bound, 1)

    """
    sr, samples = readfile(file)
    Xdb = get_stft(sr=sr, samples=samples)
    freq_rs = 93.75
    lower_bound = round(lower_bound / freq_rs)
    upper_bound = round(upper_bound / freq_rs)

    return Xdb[lower_bound:upper_bound, :]


options = read_conf()
# For Training Pool
tr_lst = options.tr_lst
htr_lst = options.htr_lst  # high frequency data train
te_lst = options.te_lst
hte_lst = options.hte_lst  # high frequency data test

# For Testing Pool
en_lst = options.en_lst
hen_lst = options.hen_lst
ve_lst = options.ve_lst
hve_lst = options.hve_lst

htr_lst = ReadList(htr_lst)
hte_lst = ReadList(hte_lst)

hen_lst = ReadList(hen_lst)
hve_lst = ReadList(hve_lst)

htr_arr = {}
hte_arr = {}
hen_arr = {}
hve_arr = {}

low_freq = 8000
high_freq = 16000
target_folder = 'enroll10-2/'

print("###############################Extract Train################\n")
for idx, htr in enumerate(htr_lst):
    print("{}/{}".format(idx, len(htr_lst)))
    htr_arr[htr] = stft2(htr, low_freq, high_freq)

print("###############################Extract Test#############\n")
for idx, hte in enumerate(hte_lst):
    print("{}/{}".format(idx, len(hte_lst)))
    hte_arr[hte] = stft2(hte, low_freq, high_freq)

print("###############################Extract Enroll#############\n")
for idx, hen in enumerate(hen_lst):
    print("{}/{}".format(idx, len(hen_lst)))
    hen_arr[hen] = stft2(hen, low_freq, high_freq)
    print(hen_arr[hen].shape)

print("###############################Extract Verify#############\n")
for idx, hve in enumerate(hve_lst):
    print("{}/{}".format(idx, len(hve_lst)))
    hve_arr[hve] = stft2(hve, low_freq, high_freq)

htr_arr = np.array(htr_arr)
hte_arr = np.array(hte_arr)
hen_arr = np.array(hen_arr)
hve_arr = np.array(hve_arr)

np.save(target_folder + "highTrFeature", htr_arr)
np.save(target_folder + "highTeFeature", hte_arr)
np.save(target_folder + "highEnFeature", hen_arr)
np.save(target_folder + "highVeFeature", hve_arr)
