# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:48:14 2016

@author: valterf
"""
from feats import stft


def print_examples(wavpaths, nnet, db_threshold=None,
                   source_intensities=None,
                   ignore_background=False,
                   pred_index=1):
    import soundfile as sf
    import numpy as np
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
        if rate != 8000:
            raise Exception("Currently only 8000 Hz audio is supported. " +
                            "You have provided a {r} Hz one".format(r=rate))
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
    from sklearn.cluster import KMeans
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
    from itertools import permutations
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

    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.imshow(img.swapaxes(0, 1), origin='lower')
    ax2.imshow(img2.swapaxes(0, 1), origin='lower')
    ax3.imshow(img3.swapaxes(0, 1), origin='lower', cmap='afmhot')
