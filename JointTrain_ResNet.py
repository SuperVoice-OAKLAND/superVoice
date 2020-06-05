# Mar.9 2020

# Difference between JointTrain2:
# This script use resNet-18 to handle high-frequency feature.

# Instead of broadcast all high frequency feature to every frame,
# This method consider get high frequency feature from each frame

import os
# import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

import sys
import numpy as np
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from dnn_models import highCNN
from data_io_apply8to16 import ReadList, read_conf, str_to_bool
from utils import *
from torchvision import models
from torchsummary import summary

resNet = models.resnet18()

def Hfeature_normalize(data):
    mu = torch.mean(data)
    std = torch.std(data)
    return (data - mu) / std


def sampleToind(beg_samp, end_samp):
    ind_beg = round(beg_samp * 12 / 512)  # Convert from sample level to STFT index level
    ind_end = round(end_samp * 12 / 512)  # 512 is hop_length of STFT, 12 is 192000/16000 = 12
    if (ind_end - ind_beg) < 75:
        ind_beg = ind_beg - 1
    if (ind_end - ind_beg) > 75:
        ind_end = ind_end - 1
    return ind_beg, ind_end


# This function embeded high frequency data reading
# hwav_lst
def create_batches_rnd(batch_size, data_folder, wav_lst, hwav_lst, N_snt, wlen, lab_dict, fact_amp):
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch = np.zeros([batch_size, wlen])
    hsig_batch = np.zeros([batch_size, 1, 86, 75])
    lab_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size=batch_size)

    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

    for i in range(batch_size):

        # select a random sentence from the list
        # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        # signal=signal.astype(float)/32768

        [signal, fs] = sf.read(data_folder + wav_lst[snt_id_arr[i]])
        # [hsignal, hfs] = sf.read(data_folder + hwav_lst[snt_id_arr[i]])

        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + wlen

        channels = len(signal.shape)
        if channels == 2:
            print('WARNING: stereo to mono: ' + data_folder + wav_lst[snt_id_arr[i]])
            signal = signal[:, 0]

        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        # print(round((snt_end*12/512)-(snt_beg*12/512)))
        high_filePath = data_folder + hwav_lst[snt_id_arr[i]]
        # hsig_frame = get_frame_freq(high_filePath, lower_bound, upper_bound, snt_beg, snt_end)
        hsig_frame = htr_feature[high_filePath]
        # print(hsig_frame.shape)
        ind_beg, ind_end = sampleToind(snt_beg, snt_end)
        hsig_frame = hsig_frame[:, ind_beg:ind_end]
        # print(hsig_frame.shape)

        # hsig_batch[i, :] = stft[8000,16000, snt_beg*12/512, snt_end*12/512]
        # (86, 75) => 86 indicates the frequency span => (16000-8000)/93.75
        #             75 indicates the time span      => (frame length 200ms => 192000*0.2=38400)/(STFT hop_length=512)
        #                                                =75
        # print(i, snt_beg, snt_end, hsig_frame.shape)

        hsig_batch[i, 0, :, :] = hsig_frame
        lab_batch[i] = lab_dict[wav_lst[snt_id_arr[i]]]

    inp = Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
    hinp = Variable(torch.from_numpy(hsig_batch).float().cuda().contiguous())
    lab = Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())

    return inp, hinp, lab


# Reading cfg file
options = read_conf()  # read cfg based on sys arg

# [data]
tr_lst = options.tr_lst
htr_lst = options.htr_lst  # high frequency data train
te_lst = options.te_lst
hte_lst = options.hte_lst  # high frequency data test
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder
output_folder = options.output_folder

# [extracted high frequency stft feature]
highFeature_Folder = 'enroll10-2/'
htr_feature = np.load(highFeature_Folder + "highTrFeature.npy")
htr_feature = htr_feature.item()
hte_feature = np.load(highFeature_Folder + "highTeFeature.npy")
hte_feature = hte_feature.item()

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

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# training list
wav_lst_tr = ReadList(tr_lst)
hwav_lst_tr = ReadList(htr_lst)
snt_tr = len(wav_lst_tr)

# test list
wav_lst_te = ReadList(te_lst)
hwav_lst_te = ReadList(hte_lst)
snt_te = len(wav_lst_te)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)  # how many samples in 200ms? 16000 samples/second, then 200ms has 16000 *0.2 = 3200
wshift = int(fs * cw_shift / 1000.00)  # 16000 *0.01 = 160

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

CNN_net = CNN(CNN_arch)  # This is the sincNet
CNN_net.cuda()

# Loading label dictionary
lab_dict = np.load(class_dict_file).item()

# highCNN for high frequency feature
highCNN_net = highCNN(resNet)
highCNN_net.cuda()

DNN1_arch = {'input_dim': CNN_net.out_dim + 100,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act,
             }
# print(CNN_net.out_dim)
# print(DNN1_arch['input_dim']) 6420
DNN1_net = MLP(DNN1_arch)
DNN1_net.cuda()
# print(DNN1_net)

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
DNN2_net.cuda()



if pt_file != 'none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
    DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])
    highCNN_net.load_state_dict(checkpoint_load['hCNN_model_par'])

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_highCNN = optim.RMSprop(highCNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

for epoch in range(N_epochs):
    # print("epoch is {}\n".format(epoch))
    test_flag = 0
    CNN_net.train()
    # highCNN_net.train()
    DNN1_net.train()
    DNN2_net.train()
    start = time.time()

    loss_sum = 0
    err_sum = 0

    for i in range(N_batches):
        # for i in range(1):

        [inp, hinp, lab] = create_batches_rnd(batch_size, data_folder, wav_lst_tr, hwav_lst_tr, snt_tr, wlen, lab_dict,
                                              0.2)

        # print(inp.shape) (128, 3200)
        # print(hinp.shape) (128, 1, 86, 75)

        out1 = CNN_net(inp)

        # print(hinp)
        # print(hinp.shape)
        hinp = Hfeature_normalize(hinp)
        # print(hinp.shape)

        # summary(highCNN_net, (1,86,75), 128)
        hout1 = highCNN_net(hinp)

        emb = torch.cat([out1, hout1], dim=1)  # (128, 100+6420=6520)
        # summary(DNN1_net, (128,6520))
        temp1 = DNN1_net(emb)

        pout = DNN2_net(temp1)
        # pout shape: 128, 13 (13 is speaker numbers in training pool)
        pred = torch.max(pout, dim=1)[1]

        loss = cost(pout, lab.long())
        err = torch.mean((pred != lab.long()).float())
        # print(i)
        if i % 10 == 0:
            with open(output_folder + "/resTrain.res", "a") as res_file:
                res_file.write("loss %f, error_count=%f\n" % (loss, sum((pred != lab.long()).float())))
            # print("pout is {}\n".format(pout))
            print("Current batch is {}".format(i))
            print("loss is {}".format(loss))
            # print("pred is {}, lable is {}\n".format(pred, lab.long()))
            print("error count is {}".format(sum((pred != lab.long()).float())))

        optimizer_CNN.zero_grad()
        optimizer_DNN1.zero_grad()
        optimizer_DNN2.zero_grad()
        # optimizer_highCNN.zero_grad()

        loss.backward()
        optimizer_CNN.step()
        # optimizer_highCNN.step()
        optimizer_DNN1.step()
        optimizer_DNN2.step()


        loss_sum = loss_sum + loss.detach()
        # print("loss sum is {}".format(loss_sum))
        err_sum = err_sum + err.detach()
        # print("err sum is {}".format(err_sum))

    loss_tot = loss_sum / N_batches
    err_tot = err_sum / N_batches

    # Full Validation  new
    # Test in same speakers, but different utterances. In our te_lst=lowbadData/test_lowbad.scp
    print(time.time()-start)
    if epoch % N_eval_epoch == 0:
        # if epoch == 0:

        CNN_net.eval()
        DNN1_net.eval()
        DNN2_net.eval()
        highCNN_net.eval()
        test_flag = 1
        loss_sum = 0
        err_sum = 0
        err_sum_snt = 0

        with torch.no_grad():
            for i in range(snt_te):
                # Each time read one sentence
                # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
                # signal=signal.astype(float)/32768

                [signal, fs] = sf.read(data_folder + wav_lst_te[i])
                # [hsignal, hfs] = sf.read(data_folder + hwav_lst_te[i])

                signal = torch.from_numpy(signal).float().cuda().contiguous()
                ########################################################################
                # hstft = getHighstft(hfs, hsignal, 80, 600)
                # hstft = hte_feature[data_folder + hwav_lst_te[i]]
                # hstft = hstft.reshape(-1, 86*75)
                # hinp = torch.from_numpy(hstft).float().cuda().contiguous()
                # hinp = Hfeature_normalize(hinp)
                ########################################################################
                hfile = data_folder + hwav_lst_te[i]
                hinp = hte_feature[hfile]

                lab_batch = lab_dict[wav_lst_te[i]]
                # lab_batch is the speaker id. Here batch size = 1

                # split signals into chunks
                beg_samp = 0
                end_samp = wlen

                # How many frames in total
                N_fr = int((signal.shape[0] - wlen) / (wshift))

                sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
                lab = Variable((torch.zeros(N_fr + 1) + lab_batch).cuda().contiguous().long())
                pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().cuda().contiguous())
                count_fr = 0
                hframe = np.zeros([Batch_dev, 1, 86 , 75])
                count_fr_tot = 0
                # 对于一个句子， 每128个frame组成一个batch，输入到网络
                # 不足128的部分， 进入下面 if count_fr > 0步骤，继续输入网络
                while end_samp < signal.shape[0]:
                    sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                    ind_beg, ind_end = sampleToind(beg_samp, end_samp)

                    hframe[count_fr, 0, :, :] = hinp[:, ind_beg:ind_end]

                    beg_samp = beg_samp + wshift
                    end_samp = beg_samp + wlen
                    count_fr = count_fr + 1
                    count_fr_tot = count_fr_tot + 1
                    # here, batch_dev 用来攒够128个frame才计算一次
                    # 把同一个句子拆分成 一个个frame，攒够128个frame 计算一次
                    if count_fr == Batch_dev:
                        inp = Variable(sig_arr)
                        out1 = CNN_net(inp)
                        hframe = Variable(torch.from_numpy(hframe).float().cuda().contiguous())
                        hframe = Hfeature_normalize(hframe)
                        hout1 = highCNN_net(hframe)
                        emb = torch.cat([out1, hout1], dim=1)
                        pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(emb))
                        count_fr = 0
                        sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
                        hframe = np.zeros([Batch_dev, 1, 86 , 75])

                if count_fr > 0:
                    inp = Variable(sig_arr[0:count_fr])
                    hframe = Variable(torch.from_numpy(hframe[0:count_fr]).float().cuda().contiguous())
                    rest_hframe = Hfeature_normalize(hframe[0:count_fr])

                    out_temp = CNN_net(inp)
                    hout = highCNN_net(rest_hframe)

                    emb = torch.cat([out_temp, hout], dim=1)
                    pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(emb))

                # In frame level, frame length is 200ms

                pred = torch.max(pout, dim=1)[1]
                loss = cost(pout, lab.long())
                # frame level err
                err = torch.mean((pred != lab.long()).float())

                # In sentence level
                print("************************Testing****************************\n")
                print(pred)
                print("************************Sum*****************************\n")
                print(torch.sum(pout, dim=0))

                [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                # 如果这个句子不是预测值，则err sentence count +1
                err_sum_snt = err_sum_snt + (best_class != lab[0]).float()
                print("Error Sentence is {}\n".format(err_sum_snt))
                print("Predict is {}\n, label is {}\n".format(best_class, lab[0]))
                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()

            # total mis-classify count ratio
            err_tot_dev_snt = err_sum_snt / snt_te
            # total loss in increment frame level
            loss_tot_dev = loss_sum / snt_te
            # frame level error for each sentence
            err_tot_dev = err_sum / snt_te

        print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
            epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        with open(output_folder + "/res.res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                      'hCNN_model_par': highCNN_net.state_dict(),
                      'DNN1_model_par': DNN1_net.state_dict(),
                      'DNN2_model_par': DNN2_net.state_dict(),
                      }
        torch.save(checkpoint, output_folder + '/model_raw.pkl')

    else:
        print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))
