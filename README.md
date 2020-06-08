# superVoice models for speaker identification and verification

This directory contains code to train and test a text-independent speaker identification and verification model.

## Prerequisites

[1] pytorch >= 1.0.0

[2] python >= 3.6

[3] librosa >= 0.7.0

[4] soundfile >= 0.10

## How to run this:
### Preparation:

1. Download datasets from **[here](https://github.com/SuperVoice-OAKLAND/datasets)**

2. Put the code fold as same parent directory of the datasets, e.g. ~/supervoice/superVoice and ~/supervoice/dataset

3. Run **downSample.py** to create another downsampled dataset for target high frequency datasets

4. Run **silence_remove.py** to remove silence audios and create _cut.wav_

5. Run **train_test_list.py** to separate __low frequency__ and __high frequency__ audios in training pool, testing pool, train audios, validation audios, enroll audios, verify audios

6. Config **cfg/SincNet_combine_apply8-16.cfg** accordingly

7. Run __extractHigh.py__ by
```
python extractHigh.py --cfg=cfg/SincNet_combine_apply8-16.cfg
```

Note to select proper high frequency range as feature

### Train:

Run __JointTrain_ResNet.py__ by

```
python JointTrain_ResNet.py --cfg=cfg/SincNet_combine_apply8-16.cfg
```

### Test:

1. Run __dvectorResNet.py__ by setting enroll_or_verify to __enroll__ to enroll speakers in testing pool, generate the enrolled embeddings

2. Run __dvectorResNet.py__ by setting enroll_or_verify to __verify__ to enroll speakers in testing pool, generate the embeddings for utterances to be verified

3. Run __liveness_detection.py__ to detect whether the utterance is made by machine or human

4. Run __computeEER.py__ by setting the enrolled embeddings and verified embeddings.
