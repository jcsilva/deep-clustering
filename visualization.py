# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:48:14 2016

@author: valterf
"""
from feats import stft, prepare_enhancement_input
from config import FRAME_RATE

from itertools import permutations

from sklearn.cluster import KMeans
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def print_examples(wavpaths, nnet, nnet_enhancement=None,
                   db_threshold=None,
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
    # Audio preparation, very similar code to feats.py generators
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
    # Audio preparation end

    # Predict embeddings
    if(isinstance(nnet.output, list)):
        V = nnet.predict(X)[pred_index]
    else:
        V = nnet.predict(X)
    x = X.reshape((-1, freq))

    # Define thresholded background as a separate class
    if db_threshold is not None:
        m = np.max(x) - db_threshold / 20.
        if ignore_background:
            v = V.reshape((-1, freq, K))
            v[x < m] = 0  # Forcing background embeddings to 0

    # KMeans clustering
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

    imgs = []
    # 1st img: Prediction
    imshape = x.shape + (3,)
    img = np.zeros((x.size, 3))
    for i in range(min(k, 3)):
        img[eg == i, i] = 1
    if db_threshold is not None:
        img[eg == kk - 1] = [0, 0, 0]
    img = img.reshape(imshape)
    imgs.append(img)

    # 2nd img: reference
    img = np.zeros((x.size, 3))
    for i in range(min(k, 3)):
        img[ref == i, i] = 1
    if db_threshold is not None:
        img[ref == kk - 1] = [0, 0, 0]
    img = img.reshape(imshape)
    imgs.append(img)

    # 3rd img: spec (with cubic power for more contrast)
    img = x
    img -= np.min(img)
    img **= 3
    imgs.append(img)

    if nnet_enhancement is not None:
        # 4th img and on: specs from enhancement net predictions
        for r in range(kk):
            E = prepare_enhancement_input(v, x, km.cluster_centers_, r)
            img = nnet_enhancement.predict(E.reshape((1,) + E.shape))
            img = img.reshape((-1, freq))
            img -= np.min(img)
            img **= 3
            imgs.append(img)

    fig, axes = plt.subplots(len(imgs), 1)
    for img, ax in zip(imgs, axes):
        ax.imshow(img.swapaxes(0, 1), origin='lower', cmap='afmhot')
