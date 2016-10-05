# Keras implementation of [Deep Clustering paper](https://arxiv.org/abs/1508.04306)

This is a keras implementation of the Deep Clustering algorithm described at https://arxiv.org/abs/1508.04306. It is not yet finished. Most of this code was implemented by [Valter Akira Miasato Filho](https://github.com/akira-miasato). 

Requirements
------------

1. System library: 
  * libsndfile1 (installed via apt-get on Ubuntu 16.04)

2. Python packages (I used Anaconda and Python 3.5):
  * Theano (pip install git+git://github.com/Theano/Theano.git)
  * keras (pip install keras)
  * pysoundfile (pip install pysoundfile)
  * numpy (conda install numpy)
  * scikit-learn (conda install scikit-learn)
  * matplotlib (conda install matplotlib)
  * python\_speech\_features (pip install python\_speech\_features)


Training the network
--------------------

First of all, you must create two text files: train\_list and valid\_list. They must contain your training and validation data. The lines of these files must be according to the following pattern:
```
path/to/audioFile1 spk1
path/to/audioFile2 spk2
path/to/audioFile3 spk1
```
spk1, spk2 identifies the speaker that uttered the recorded sentence. 


The current implementation works with audio files recorded at 16 kHz and 8 kHz. It was already tested with flac and wav files, but it should work with all formats supported by [pysoundfile/libsndfile](http://www.mega-nerd.com/libsndfile/#Features).


After creating train\_list and valid\_list, you may start training the network with the command:
```
python keras-speech-separation-batch.py
```

References
----------
* https://arxiv.org/abs/1508.04306
