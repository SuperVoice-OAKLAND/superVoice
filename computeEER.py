# This script compute EER by giving speaker embeddings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import time

num_speaker = 4
enroll_d_vec = "d_vectors/enroll_enroll10-1.npy"
verify_d_vec = "d_vectors/verify_enroll10-1.npy"

en_embs = np.load(enroll_d_vec, allow_pickle=True)
ve_embs = np.load(verify_d_vec, allow_pickle=True)
# print(len(ve_embs.item().keys()))
speaker_emb = {}
ve_speaker_emb = {}
for key, value in en_embs.item().items():
    speakerId = int(key.split("/")[-3])
    if speakerId not in speaker_emb.keys():
        speaker_emb[speakerId] = value
        continue
    speaker_emb[speakerId] = np.vstack((speaker_emb[speakerId], value))

# print(speaker_emb[14].shape)

for key, value in ve_embs.item().items():
    ve_speakerId = int(key.split("/")[-3])
    if ve_speakerId not in ve_speaker_emb.keys():
        ve_speaker_emb[ve_speakerId] = value
        continue
    ve_speaker_emb[ve_speakerId] = np.vstack((ve_speaker_emb[ve_speakerId], value))

# print(ve_speaker_emb[14].shape)

err_count = 0   # speaker identification ERROR rate
ve_count = len(ve_embs.item())

print(ve_count)

for ve_speaker, ve_utters in ve_speaker_emb.items():
    best = {}
    for en_speaker, en_utters in speaker_emb.items():
        sim = cosine_similarity(ve_utters, en_utters)
        for idx, utter_test in enumerate(sim):
            if idx not in best.keys():
                best[idx] = {}
                best[idx]['testSpeaker'] = ve_speaker
                best[idx]['similarity'] = np.max(utter_test)
                best[idx]['from'] = en_speaker
            if np.max(utter_test) > best[idx]['similarity']:
                best[idx]['similarity'] = np.max(utter_test)
                best[idx]['from'] = en_speaker

    for k, v in best.items():
        if v['testSpeaker'] != v['from']:
            err_count = err_count + 1

# print(ve_count)
err_rate = err_count*100/ve_count
print("ERR COUNT IS : {0}\nERR RATE IS : {1:.2f}%\n".format(err_count, err_rate))

threshold = 0
false_accept = 0
false_reject = 0
FARs = []
FRRs = []
thes = []

while threshold < 1:
    false_accept = 0
    false_reject = 0
    for ve_speaker, ve_utters in ve_speaker_emb.items():
        for en_speaker, en_utters in speaker_emb.items():
            start = time.time()
            sim = cosine_similarity(ve_utters, en_utters)
            # sim = (sim + 1) / 2
            sim = np.mean(sim, axis=1)
            mask = sim - threshold
            close = (mask > 0).sum()
            far = (mask < 0).sum()
            if ve_speaker == en_speaker:
                false_reject += far
            else:
                false_accept += close
            print(time.time()-start)
    FAR = 100*false_accept / (ve_count*num_speaker)
    FRR = 100*false_reject / (ve_count*num_speaker)
    FARs.append(FAR)
    FRRs.append(FRR)
    threshold = threshold + 0.01
    thes.append(threshold)

mindis = 100
npFARs = np.array(FARs)
npFRRs = np.array(FRRs)
inter = np.argmin(abs(npFARs - npFRRs))
# print(inter)
ERR = npFARs[inter]

print("EER IS : {0:.2f}%\nBEST THRESHOLD IS : {1:.2f}\n".format(ERR, 0.01*inter))

FARs = np.array(FARs)
FRRs = np.array(FRRs)
thes = np.array(thes)
scipy.io.savemat('far.mat', {'far':FARs})
scipy.io.savemat('frr.mat', {'frr':FRRs})
scipy.io.savemat('thes.mat', {'theshold':thes})

plt.plot(thes, FARs, marker='o', label="FAR", markerfacecolor='blue', markersize=2, color='skyblue', linewidth=4)
plt.plot(thes, FRRs, marker='', label="FRR", color='olive', linewidth=2)
plt.legend()
plt.xlabel("Threshold")
plt.ylabel("%")
plt.tick_params(labelright=True)
plt.title('Sinc-32to48kEER')
plt.show()



