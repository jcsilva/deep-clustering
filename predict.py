# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:20:31 2016

@author: valterf
"""
import soundfile as sf
import numpy as np
from feats import stft, istft
from config import FRAME_RATE


def prepare_features(wavpath, nnet, pred_index=0):
    freq = int(nnet.input.get_shape()[2])
    if(isinstance(nnet.output, list)):
        K = int(nnet.output[pred_index].get_shape()[2]) // freq
    else:
        K = int(nnet.output.get_shape()[2]) // freq
    sig, rate = sf.read(wavpath)
    if rate != FRAME_RATE:
        raise Exception("Config specifies " + str(FRAME_RATE) +
                        "Hz as sample rate, but file " + str(wavpath) +
                        "is in " + str(rate) + "Hz.")
    sig = sig - np.mean(sig)
    sig = sig/np.max(np.abs(sig))
    spec = stft(sig)
    mag = np.real(np.log10(spec))
    X = mag.reshape((1,) + mag.shape)
    if(isinstance(nnet.output, list)):
        V = nnet.predict(X)[pred_index]
    else:
        V = nnet.predict(X)

    x = X.reshape((-1, freq))
    v = V.reshape((-1, K))

    return spec, rate, x, v


def separate_sources(wavpath, nnet, num_sources, out_prefix):
    """
    Separates sources from a single-channel multiple-source input.

    wavpath is the path for the mixed input

    nnet is a loaded pre-trained model using the "load_model" function from
    predict.py

    num_sources is the expected number of sources from the input, and defines
    the number of output files

    out_prefix is the prefix of each output file, which will be writtin on the
    form {prefix}-N.wav, N in [0..num_sources-1]
    """
    k = num_sources
    freq = int(nnet.input.get_shape()[2])
    spec, rate, x, v = prepare_features(wavpath, nnet, 1)

    from sklearn.cluster import KMeans
    km = KMeans(k)
    eg = km.fit_predict(v)

    imgs = np.zeros((k, eg.size))
    for i in range(k):
        imgs[i, eg == i] = 1

    spec = np.log(spec)
    mag = np.real(spec)
    phase = np.imag(spec)
    i = 1
    for img in imgs:
        mask = img.reshape(-1, freq)
        sig_out = istft(np.exp(mag + 1j * phase) * mask)
        sig_out -= np.mean(sig_out)
        sig_out /= np.max(sig_out)
        sf.write(out_prefix + '_{}.wav'.format(i), sig_out, rate)
        i += 1
