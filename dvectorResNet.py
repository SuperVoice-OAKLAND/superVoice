

import os
import soundfile as sf
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from dnn_models import MLP
from dnn_models import SincNet as CNN
from dnn_models import highCNN
from data_io_apply8to16 import ReadList, read_conf_inp, str_to_bool
import sys
from utils import *
from torchvision import models
import time

def sampleToind(beg_samp, end_samp):
    ind_beg = round(beg_samp * 12 / 512)  # Convert from sample level to STFT index level
    ind_end = round(end_samp * 12 / 512)  # 512 is hop_length of STFT, 12 is 192000/16000 = 12
    if (ind_end - ind_beg) < 75:
        ind_beg = ind_beg - 1
        # print("less than 75: {}".format(ind_end - ind_beg))
    if (ind_end - ind_beg) > 75:
        ind_end = ind_end - 1
    #     print("greater than 75: {}".format(ind_end - ind_beg))
    # print(ind_beg, ind_end)
    return ind_beg, ind_end


folder = 'enroll10-2'
model_folder = folder
setting_folder = folder
model_identifier = folder

cfg = 'SincNet_combine_apply8-16.cfg'

enroll_or_verify = 'enroll'
global_path = '/home/hanqing/1ASpeakerRecognition/SincNet/'
# Model to use for computing the d-vectors
model_file = global_path + 'exp/'+model_folder+'/model_raw.pkl'
cfg_file = global_path + 'cfg/'+cfg


# model_file = '/home/hanqing/1ASpeakerRecognition/SincNet/exp/SincNet_joint_norm_new/model_raw.pkl'  # This is the model to use for computing the d-vectors (it should be pre-trained using the speaker-id DNN)
# cfg_file = '/home/hanqing/1ASpeakerRecognition/SincNet/cfg/SincNet_combine_apply8-16.cfg'  # Config file of the speaker-id experiment used to generate the model
# convert which speakers to d_vector?
# enroll_lowbad.scp
# Or
# verify_highbad.scp
te_lst = global_path+setting_folder+'/'+enroll_or_verify+'_joint_low.scp'
hte_lst = global_path+setting_folder+'/'+enroll_or_verify+'_joint_high.scp'

# te_lst = '/home/hanqing/1ASpeakerRecognition/SincNet/Low_8to16k/verify_joint_low.scp'  # List of the wav files to process
# hte_lst = '/home/hanqing/1ASpeakerRecognition/SincNet/Low_8to16k/verify_joint_high.scp' # List of the high frequency process

out_dict_file = './d_vectors/'+enroll_or_verify+'_'+model_identifier
# out_dict_file = './d_vectors/d_vect_joint_verify_norm_new.npy'  # output dictionary containing the a sentence id as key as the d-vector as value


avoid_small_en_fr = True
energy_th = 0.1  # Avoid frames with an energy that is 1/10 over the average energy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = None

# Reading cfg file
print(cfg_file)
options = read_conf_inp(cfg_file)

# [data]
data_folder = options.data_folder + '/'
pt_file = options.pt_file
output_folder = options.output_folder

h_feature = np.load(setting_folder+ "/highEnFeature.npy" if enroll_or_verify=='enroll'
                    else setting_folder+ "/highVeFeature.npy")  # here either highEn or highVe
h_feature = h_feature.item()

# [extracted high frequency stft feature]
# h_feature = np.load("highVeFeature.npy")  # here either highEn or highVe
# h_feature = h_feature.item()
# hte_feature = np.load("highTeFeature.npy")
# hte_feature = hte_feature.item()


# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

# [dnn]
fc_lay = list(map(int, options.fc_lay.split(',')))
fc_drop = list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act = list(map(str, options.fc_act.split(',')))

# [class]
class_lay = list(map(int, options.class_lay.split(',')))
class_drop = list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act = list(map(str, options.class_act.split(',')))

wav_lst_te = ReadList(te_lst)
hwav_lst_te = ReadList(hte_lst)

snt_te = len(wav_lst_te)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

# loss function
cost = nn.NLLLoss()

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = 128

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop,
            }

CNN_net = CNN(CNN_arch)
CNN_net.to(device)

resNet = models.resnet18()
highCNN_net = highCNN(resNet)
highCNN_net.to(device)

DNN1_arch = {'input_dim': CNN_net.out_dim + 100,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act,
             }

DNN1_net = MLP(DNN1_arch)
DNN1_net.to(device)

DNN2_arch = {'input_dim': fc_lay[-1],
             'fc_lay': class_lay,
             'fc_drop': class_drop,
             'fc_use_batchnorm': class_use_batchnorm,
             'fc_use_laynorm': class_use_laynorm,
             'fc_use_laynorm_inp': class_use_laynorm_inp,
             'fc_use_batchnorm_inp': class_use_batchnorm_inp,
             'fc_act': class_act,
             }

DNN2_net = MLP(DNN2_arch)
DNN2_net.to(device)

checkpoint_load = torch.load(model_file)
CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])
highCNN_net.load_state_dict(checkpoint_load['hCNN_model_par'])

CNN_net.eval()
DNN1_net.eval()
DNN2_net.eval()
test_flag = 1

d_vector_dim = fc_lay[-1]
d_vect_dict = {}

def Hfeature_normalize(data):
    mu = torch.mean(data)
    std = torch.std(data)
    return (data - mu)/std

with torch.no_grad():
    for i in range(snt_te):
        start = time.time()
        [signal, fs] = sf.read(data_folder + '/' + wav_lst_te[i])

        # Amplitude normalization
        signal = signal / np.max(np.abs(signal))

        signal = torch.from_numpy(signal).float().to(device).contiguous()

        #############################################################################
        # High frequency Feature
        #############################################################################
        hinp = h_feature[hwav_lst_te[i]]

        # hstft = h_feature[hwav_lst_te[i]]
        # hstft = hstft.reshape(-1, 520)
        # hinp = torch.from_numpy(hstft).float().cuda().contiguous()
        # hinp = Hfeature_normalize(hinp)

        if avoid_small_en_fr:
            # computing energy on each frame:
            beg_samp = 0
            end_samp = wlen

            N_fr = int((signal.shape[0] - wlen) / (wshift))
            Batch_dev = N_fr
            en_arr = torch.zeros(N_fr).float().contiguous().to(device)
            count_fr = 0
            count_fr_tot = 0
            while end_samp < signal.shape[0]:
                en_arr[count_fr] = torch.sum(signal[beg_samp:end_samp].pow(2))
                beg_samp = beg_samp + wshift
                end_samp = beg_samp + wlen
                count_fr = count_fr + 1
                count_fr_tot = count_fr_tot + 1
                if count_fr == N_fr:
                    break

            en_arr_bin = en_arr > torch.mean(en_arr) * 0.1
            en_arr_bin.to(device)
            n_vect_elem = torch.sum(en_arr_bin)

            if n_vect_elem < 10:
                print('only few elements used to compute d-vectors')
                sys.exit(0)

        # split signals into chunks
        beg_samp = 0
        end_samp = wlen

        N_fr = int((signal.shape[0] - wlen) / (wshift))

        sig_arr = torch.zeros([Batch_dev, wlen]).float().to(device).contiguous()
        hframe = np.zeros([Batch_dev, 1, 86 , 75])
        dvects = Variable(torch.zeros(N_fr, d_vector_dim).float().to(device).contiguous())
        count_fr = 0
        count_fr_tot = 0
        while end_samp < signal.shape[0]:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            ind_beg, ind_end = sampleToind(beg_samp, end_samp)
            while(ind_end > hinp.shape[1]):
                ind_beg = ind_beg -1
                ind_end = ind_end -1
            hframe[count_fr, 0, :, :] = hinp[:, ind_beg:ind_end]

            beg_samp = beg_samp + wshift
            end_samp = beg_samp + wlen
            count_fr = count_fr + 1
            count_fr_tot = count_fr_tot + 1
            if count_fr == Batch_dev:
                inp = Variable(sig_arr)
                out1 = CNN_net(inp)
                hframe = Variable(torch.from_numpy(hframe).float().cuda().contiguous())
                hframe = Hfeature_normalize(hframe)
                hout1 = highCNN_net(hframe)
                emb = torch.cat([out1, hout1], dim=1)
                dvects[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN1_net(emb)
                count_fr = 0
                sig_arr = torch.zeros([Batch_dev, wlen]).float().to(device).contiguous()
                hframe = np.zeros([Batch_dev, 1, 86 , 75])

        if count_fr > 0:
            inp = Variable(sig_arr[0:count_fr])
            hframe = Variable(torch.from_numpy(hframe[0:count_fr]).float().cuda().contiguous())
            rest_hframe = Hfeature_normalize(hframe[0:count_fr])
            hout = highCNN_net(rest_hframe)
            out_temp = CNN_net(inp)
            emb = torch.cat([out_temp, hout], dim=1)
            dvects[count_fr_tot - count_fr:count_fr_tot, :] = DNN1_net(emb)

        if avoid_small_en_fr:
            dvects = dvects.index_select(0, (en_arr_bin == 1).nonzero().view(-1))

        # averaging and normalizing all the d-vectors
        d_vect_out = torch.mean(dvects / dvects.norm(p=2, dim=1).view(-1, 1), dim=0)
        print(time.time()-start)

        # checks for nan
        nan_sum = torch.sum(torch.isnan(d_vect_out))

        if nan_sum > 0:
            print(wav_lst_te[i])
            sys.exit(0)

        # saving the d-vector in a numpy dictionary
        #dict_key = wav_lst_te[i].split('/')[-2] + '/' + wav_lst_te[i].split('/')[-1]
        dict_key = wav_lst_te[i]
        d_vect_dict[dict_key] = d_vect_out.cpu().numpy()
        print(dict_key)

# Save the dictionary
np.save(out_dict_file, d_vect_dict)
