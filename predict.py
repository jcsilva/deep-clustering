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
    v = V.reshape((-1, freq))

    return spec, rate, x, v


def denoise(wavpath, nnet, out):
    """
    """
    spec, rate, x, v = prepare_features(wavpath, nnet, 1)

#    from matplotlib import pyplot as plt
#    plt.imshow(v.swapaxes(0, 1), origin='lower')
#    plt.colorbar()

    spec = np.log(spec)
    mag = np.real(spec) - v * np.log(10)
    phase = np.imag(spec)
    i = 1
    sig_out = istft(np.exp(mag + 1j * phase))
    sig_out -= np.mean(sig_out)
    sig_out /= np.max(sig_out)
    sf.write(out, sig_out, rate)
    i += 1
