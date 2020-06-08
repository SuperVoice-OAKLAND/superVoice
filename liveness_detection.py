import librosa.display
import scipy
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import scipy


def showSpec(file, n_fft=2048, win_length=2048, hop_length=512):
    #   start_time = time.time()
    sr, sp = scipy.io.wavfile.read(file)

    X = librosa.stft(sp.astype('float'), n_fft=2048, hop_length=512)
    #     print("--- load use %s seconds ---" % (time.time() - start_time))
    Xdb = librosa.amplitude_to_db(np.abs(X))
    print(Xdb.shape)
    librosa.display.specshow(Xdb, sr=sr, x_axis='frames', y_axis='hz')
    plt.show()
    return Xdb


def freq_trend(Xdb):
    sum_row = []
    for row in range(Xdb.shape[0]):
        sum_row.append(sum(Xdb[row]))
    sum_row = np.array(sum_row)
    sum_row = np.array(sum_row)
    sum_row = sum_row - np.mean(sum_row)
    high_ratio = sum(sum_row[256:600]) / sum(sum_row[0:600])
    return high_ratio


def determine(file):
    start_time = time.time()
    Xdb = showSpec(file)
    high_ratio = freq_trend(Xdb)
    if high_ratio > -0.5:
        print('This is a human')
        return 0
    else:
        print('This is a Machine')
        return 1


if __name__ == '__main__':
    file = ""
    determine(file)
