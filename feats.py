# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
STFT/ISTFT derived from Basj's implementation[1], with minor modifications,
such as the replacement of the hann window by its root square, as specified in
the original paper from Hershey et. al. (2015)[2]

[1] http://stackoverflow.com/a/20409020
[2] https://arxiv.org/abs/1508.04306
"""
import numpy as np
import random
import soundfile as sf
from copy import deepcopy
from config import FRAME_LENGTH, FRAME_SHIFT, FRAME_RATE
from config import TIMESTEPS, DB_THRESHOLD


def sqrt_hann(M):
    return np.sqrt(np.hanning(M))


def stft(x, fftsize=int(FRAME_LENGTH*FRAME_RATE),
         overlap=FRAME_LENGTH//FRAME_SHIFT):
    """
    Short-time fourier transform.
        x:
        input waveform (1D array of samples)

        fftsize:
        in samples, size of the fft window

        overlap:
        should be a divisor of fftsize, represents the rate of
        window superposition (window displacement=fftsize/overlap)

        return: linear domain spectrum (2D complex array)
    """
    hop = int(np.round(fftsize / overlap))
    w = sqrt_hann(fftsize)
    out = np.array([np.fft.rfft(w*x[i:i+fftsize])
                    for i in range(0, len(x)-fftsize, hop)])
    return out


def istft(X, overlap=FRAME_LENGTH//FRAME_SHIFT):
    """
    Inverse short-time fourier transform.
        X:
        input spectrum (2D complex array)

        overlap:
        should be a divisor of (X.shape[1] - 1) * 2, represents the rate of
        window superposition (window displacement=fftsize/overlap)

        return: floating-point waveform samples (1D array)
    """
    fftsize = (X.shape[1] - 1) * 2
    hop = int(np.round(fftsize / overlap))
    w = sqrt_hann(fftsize)
    x = np.zeros(X.shape[0]*hop)
    wsum = np.zeros(X.shape[0]*hop)
    for n, i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += np.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x


def mix_noise(bg_data, spk_data, min_snr=5.0, max_snr=20.0, min_delta=0.001):

    # From dB to linear scale
    min_dba = np.sqrt(10.**(min_snr/10.))
    max_dba = np.sqrt(10.**(max_snr/10.))

    # Calculates the histogram from the signal, counts all occurences
    # of deltas below the minimum threshold, discards them from the
    # final count
    deltas = np.abs(spk_data[1:] - spk_data[:-1])
    max_d = np.max(deltas)
    vad = np.histogram(np.abs(spk_data[1:] - spk_data[:-1]),
                       bins=[
                       0,
                       max_d*min_delta,
                       float('Inf')
                       ])[0][1]
    spk_en = np.sum(np.square(spk_data)) / vad

    # Samples a random sequence from the bg noise waveform
    bg_en = 0.0
    q = 0.0
    while(bg_en <= 0 or q == 0):
        beg = random.randint(0, len(bg_data) - len(spk_data))
        bg_en = np.mean(np.square(bg_data[beg:beg + len(spk_data)]))
        q = random.uniform(min_dba, max_dba)

    # Superposition with a random ratio from the snr range
    ratio = np.sqrt(abs(spk_en / bg_en)) / q
    noisy = spk_data + bg_data[beg:beg + len(spk_data)] * ratio

    return noisy


def get_egs(speechlist, noiselist, batch_size=1):
    """
    Generate examples for the neural network from a list of clean speech
    and noise audio files
    """
    speech_paths = []
    noise_paths = []
    batch_x = []
    batch_y = []
    batch_count = 0

    def get_logspec(x):
        return np.log10(np.absolute(stft(x)) + 1e-7)

    while True:  # Generate examples indefinitely
        for paths, wavlist in [(speech_paths, speechlist),
                               (noise_paths, noiselist)]:
            if len(paths) == 0:
                # Reading wav lists
                f = open(wavlist)
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    paths.append(line)
                f.close()
                # Randomizing wav lists
                random.shuffle(paths)
        sigs = []

        # Pop wav files from speech and noisy datasets and mix them
        sigs = []
        for paths in [speech_paths, noise_paths]:
            p = paths.pop()
            sig, rate = sf.read(p)
            if rate != FRAME_RATE:
                raise Exception("Config specifies " + str(FRAME_RATE) +
                                "Hz as sample rate, but file " + str(p) +
                                "is in " + str(rate) + "Hz.")
            sig = sig - np.mean(sig)
            sig = sig/np.max(np.abs(sig))
            sigs.append(deepcopy(sig))
        clean, bg_data = sigs
        noisy = mix_noise(bg_data, clean)

        # Get input and output
        X = get_logspec(noisy)
        Y = get_logspec(clean)
        Y = X - Y
        if len(X) <= TIMESTEPS:
            continue

        # Generating sequences
        i = 0
        while i + TIMESTEPS < len(X):
            batch_x.append(X[i:i+TIMESTEPS])
            batch_y.append(Y[i:i+TIMESTEPS])
            i += TIMESTEPS//2

            batch_count = batch_count+1

            if batch_count == batch_size:
                inp = np.array(batch_x).reshape((batch_size,
                                                 TIMESTEPS, -1))
                out = np.array(batch_y).reshape((batch_size,
                                                 TIMESTEPS, -1))
                yield({'input': inp},
                      {'irm': out})
                batch_x = []
                batch_y = []
                batch_count = 0


if __name__ == "__main__":
    x, y = next(get_egs('wavlist', 'noisel', batch_size=50))
    print(x['input'].shape)
    print(y['irm'].shape)
