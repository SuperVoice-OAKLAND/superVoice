import sox
import os
import glob

tfm = sox.Transformer()
tfm.convert(samplerate=16000)
curdir = os.getcwd()
downsampled = "lowfrequency"
datasetPath = "../oakland-dataset"
dataset = "dataset_1"

originalDataset = os.path.join(datasetPath, dataset)
downsampleDataset = os.path.join(datasetPath, downsampled)
if not os.path.exists(downsampleDataset):
    os.mkdir(downsampleDataset)

for speaker in range(len(os.listdir(originalDataset))):
    if not os.path.exists(os.path.join(downsampleDataset, str(speaker+1))):
        os.mkdir(os.path.join(downsampleDataset, str(speaker+1)))

audios = glob.glob(originalDataset+"/*/*.wav")
print(audios)
for audio in audios:
    output = audio.split('/')
    output[2] = downsampled
    opath = "/".join(output)
    print(opath)
    tfm.build(audio, opath)

