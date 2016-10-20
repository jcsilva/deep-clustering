# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:48:14 2016

@author: valterf
"""
from feats import stft
from config import FRAME_RATE

from itertools import permutations

from sklearn.cluster import KMeans
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def print_examples(wavpaths, nnet, db_threshold=None,
                   source_intensities=None,
                   ignore_background=False,
                   pred_index=0):
    """
    Visualizes the difference between the neural network output and the ground
    truth from a list of files to be mixed up.

    wavpaths is a list of wavfile paths which should all have sampling rate.

    nnet is a loaded pre-trained model using the "load_model" function from
    predict.py

    db_threshold is the backgorund silence threshold, if you wish to treat
    background silence as a "different source"

    source_intensities is a list of floats which weights the source intensities
    of each file from wavpaths by multiplying its amp. It should have the same
    length of wavpaths

    ignore_background is used only if db_threshold is specified. It forces all
    background silence points to the same region, forcing the KMeans algorithm
    to treat this point as a centroid and as a cluster.

    pred_index is a parameter for multiple output networks, and should be
    ignored for now.
    """
    k = len(wavpaths)
    kk = k
    if db_threshold is not None:
        kk += 1
    freq = int(nnet.input.get_shape()[2])
    if(isinstance(nnet.output, list)):
        K = int(nnet.output[pred_index].get_shape()[2]) // freq
    else:
        K = int(nnet.output.get_shape()[2]) // freq
    sigsum = None
    specs = []
    sigs = []
    for i, wavpath in enumerate(wavpaths):
        sig, rate = sf.read(wavpath)
        if rate != FRAME_RATE:
            raise Exception("Config specifies " + str(FRAME_RATE) +
                            "Hz as sample rate, but file " + str(wavpath) +
                            "is in " + str(rate) + "Hz.")
        sig = sig - np.mean(sig)
        sig = sig/np.max(np.abs(sig))
        if source_intensities is not None:
            sig *= source_intensities[i]
        if sigsum is None:
            sigsum = sig
        else:
            sigsum = sigsum[:len(sig)] + sig[:len(sigsum)]
        sigs.append(sig)
    for sig in sigs:
        specs.append(np.real(stft(sig[:len(sigsum)], rate)))
    specs = np.transpose(np.array(specs), (1, 2, 0))
    sigsum = sigsum - np.mean(sigsum)
    sigsum = sigsum/np.max(np.abs(sigsum))
    mag = np.real(stft(sigsum, rate))
    X = mag.reshape((1,) + mag.shape)
    if(isinstance(nnet.output, list)):
        V = nnet.predict(X)[pred_index]
    else:
        V = nnet.predict(X)
    x = X.reshape((-1, freq))
    if db_threshold is not None:
        v = V.reshape((-1, freq, K))
        m = np.max(x) - db_threshold / 20.
        if ignore_background:
            v[x < m] = 0
    v = V.reshape((-1, K))
    km = KMeans(kk)
    eg = km.fit_predict(v)
    ref = np.argmax(specs.reshape((x.size, k)), axis=1)
    if db_threshold is not None:
        ref[(x < m).reshape(ref.shape)] = kk - 1

    # Permute classes for oracle alignment
    eg_p = np.zeros((x.size, kk))
    ref_p = np.zeros((x.size, kk))
    for i in range(kk):
        eg_p[eg == i, i] = 1
        ref_p[ref == i, i] = 1
    p = None
    s = np.float('Inf')
    for pp in permutations(list(range(kk))):
        ss = np.sum(np.abs(ref_p - eg_p[:, pp]))
        if ss < s:
            s = ss
            p = pp
    eg_p = eg_p[:, p]
    eg = np.argmax(eg_p, axis=1)

    imshape = x.shape + (3,)
    img = np.zeros((x.size, 3))
    for i in range(min(k, 3)):
        img[eg == i, i] = 1
    if db_threshold is not None:
        img[eg == kk - 1] = [0, 0, 0]
    img = img.reshape(imshape)

    img2 = np.zeros((x.size, 3))
    for i in range(min(k, 3)):
        img2[ref == i, i] = 1
    if db_threshold is not None:
        img2[ref == kk - 1] = [0, 0, 0]
    img2 = img2.reshape(imshape)

    img3 = x
    img3 -= np.min(img3)
    img3 **= 3

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.imshow(img.swapaxes(0, 1), origin='lower')
    ax2.imshow(img2.swapaxes(0, 1), origin='lower')
    ax3.imshow(img3.swapaxes(0, 1), origin='lower', cmap='afmhot')
