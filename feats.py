# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""
import numpy as np
import random
import scipy.io.wavfile as wav
from python_speech_features import sigproc

FRAME_LENGTH = .032
FRAME_SHIFT = .008
FS = 8000
CONTEXT = 100


def squared_hann(M):
    return np.sqrt(np.hanning(M))


def stft(sig, rate):
    frames = sigproc.framesig(sig,
                              FRAME_LENGTH*rate,
                              FRAME_SHIFT*rate,
                              winfunc=squared_hann)
    spec = np.fft.rfft(frames, int(FRAME_LENGTH*rate))
    return np.log(np.absolute(spec))


def get_batches(min_mix=3, max_mix=3):
    wavs = []
    while True:
        # Select number of files to mix
        k = np.random.randint(min_mix, max_mix+1)
        if(k > len(wavs)):
            # Reading wav files list and randomizing inputs
            wavs = []
            f = open('wavlist')
            for line in f:
                wavs.append(line.strip())
            f.close()
            random.shuffle(wavs)
        wavsum = None
        sigs = []

        # Read selected wav files, store them individually and mix them
        for i in range(k):
            p = wavs.pop()
            rate, sig = wav.read(p)
            sig = sig - np.mean(sig)
            sig = sig/np.max(np.abs(sig))
            sig *= (np.random.random()*3/4 + 1/4)
            if wavsum is None:
                wavsum = sig
            else:
                wavsum = wavsum[:len(sig)] + sig[:len(wavsum)]
            sigs.append(sig)

        # STFT for mixed signal
        X = np.real(stft(wavsum, rate))
        if len(X) <= CONTEXT:
            continue

        # STFTs for individual files
        specs = []
        for sig in sigs:
            specs.append(stft(sig[:len(wavsum)], rate))
        specs = np.array(specs)

        # Mask for ditching silence components
        Y = np.zeros(X.shape + (k,))

        # Get dominant spectra indexes, create one-hot outputs
        m = np.max(X) - 5.5
        vals = np.argmax(specs, axis=0)
        for i in range(k):
            t = np.zeros(k)
            t[i] = 1
            Y[vals == i] = t

        # Create mask for zeroing out gradients from silence components
        m = np.max(X) - 5.5
        M = np.ones(X.shape)
        M[X < m] = [0]
#        Y[X < m] = np.zeros(k)
        i = 0

        # Generating batches
        while i + CONTEXT < len(X):
            yield(X[i:i+CONTEXT].reshape((1, 100, 129)),
                  Y[i:i+CONTEXT].reshape((1, -1)))
            i += CONTEXT // 2


if __name__ == "__main__":
    a = get_batches()
    for i,j in a:
        print(i.shape,j.shape)