import glob
import random
import numpy as np
import os

########################################################################################
# Speaker 3-12 as training speaker pool,
# Speaker 13-16 as test speaker pool.
# Here split train and test set within training pool.
# For example, speaker 3 has 25 utterances, we randomly choose 18 sentences to train,
# the rest of 7 sentences will be used to test.
########################################################################################

def mapTohigh(lowscp):
    output = lowscp.split('/')
    c = 'HIGH'+output[1][3:]
    output[1] = c
    high = "/".join(output)
    f_high = open(high, 'w')
    with open(lowscp) as f:
        lines = f.readlines()
    for line in lines:
        # print(line.split('/'))
        olist = line.split('/')
        olist[-3] = "dataset_1"
        olist[-1] = olist[-1][:-8] + "high" + "cut.wav\n"
        npath = "/".join(olist)
        f_high.write(npath)
    f_high.close()

if __name__ == '__main__':
    files = glob.glob("../oakland-dataset/lowfrequency/*/*cut.wav")
    trainpool_train = "prepare/LOW_trainpool_train.scp"
    trainpool_validation = "prepare/LOW_trainpool_validation.scp"
    testpool_enroll = "prepare/LOW_testpool_enroll.scp"
    testpool_verify = "prepare/LOW_testpool_verify.scp"

    ftrain = open(trainpool_train, "w")
    ftest = open(trainpool_validation, "w")
    train_sent = 5
    test_sent = 20
    enroll_sent = 3
    fenroll = open(testpool_enroll, "w")
    fverify = open(testpool_verify, "w")
    #######################################################################################
    # train_pool contains train_utter and test_utter.
    # They will use to training procedue (Speaker_id.py)
    #######################################################################################
    train_pool = {}
    train_utter = {}
    test_utter = {}
    labels = {}
    #######################################################################################
    # test_pool contains enroll_utter and test_enroll_utter.
    # Speakers in test_pool is those the network never seen before
    #######################################################################################
    test_pool = {}
    enroll_utter = {}
    test_enroll_utter = {}
    test_labels = {}

    for file in files:
        speakerId = int(file.split("/")[-2])

        if speakerId < 11:  # Means in training speaker pool
            if speakerId not in train_pool.keys():
                train_pool[speakerId] = []
            train_pool[speakerId].append(file)
        else:
            if speakerId not in test_pool.keys():
                test_pool[speakerId] = []
            test_pool[speakerId].append(file)

    # Generate all train and test labels and save to .npy
    for k, v in train_pool.items():
        random.shuffle(v)
        train_utter[k] = v[:train_sent]
        test_utter[k] = v[train_sent:]

    for k, v in train_utter.items():
        for utter in v:
            labels[utter] = k
            ftrain.write(utter+"\n")

    for k, v in test_utter.items():
        for utter in v:
            labels[utter] = k
            ftest.write(utter+"\n")

    npLab = np.array(labels)
    np.save('prepare/LOW_trainpool_labels.npy', npLab)


    for k, v in test_pool.items():
        random.shuffle(v)
        enroll_utter[k] = v[:enroll_sent]
        test_enroll_utter[k] = v[enroll_sent:]

    for k, v in enroll_utter.items():
        for utter in v:
            test_labels[utter] = k
            fenroll.write(utter+"\n")

    for k, v in test_enroll_utter.items():
        for utter in v:
            test_labels[utter] = k
            fverify.write(utter+"\n")

    npEnrollLab = np.array(test_labels)
    np.save('prepare/LOW_testpool_labels.npy', npEnrollLab)
    ftrain.close()
    ftest.close()
    fenroll.close()
    fverify.close()
    mapTohigh(trainpool_train)
    mapTohigh(trainpool_validation)
    mapTohigh(testpool_enroll)
    mapTohigh(testpool_verify)