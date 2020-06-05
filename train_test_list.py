import glob
import random
import numpy as np

########################################################################################
# Speaker 3-12 as training speaker pool,
# Speaker 13-16 as test speaker pool.
# Here split train and test set within training pool.
# For example, speaker 3 has 25 utterances, we randomly choose 18 sentences to train,
# the rest of 7 sentences will be used to test.
########################################################################################


if __name__ == '__main__':
    files = glob.glob("/home/hanqing/1ASpeakerRecognition/ultraData/*/bad/*cut.wav")
    ftrain = open("highbad/train15_highbad.scp", "w")
    ftest = open("highbad/test7_highbad.scp", "w")
    train_sent = 18
    test_sent = 7
    enroll_sent = 3
    fenroll = open("highbad/enroll3_highbad.scp", "w")
    fverify = open("highbad/verify22_highbad.scp", "w")
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
        speakerId = int(file.split("/")[-3])

        if speakerId < 13:  # Means in training speaker pool
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
    np.save('highbad/highbad.npy', npLab)


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
    np.save('highbad/highbad3_verify.npy', npEnrollLab)
