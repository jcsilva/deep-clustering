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
from config import FRAME_RATE, FRAME_LENGTH, FRAME_SHIFT, TIMESTEPS


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


def get_egs(wavlist, min_mix=2, max_mix=3, batch_size=1, noiselist=None):
    """
    Generate examples for the neural network from a list of wave files with
    speaker ids. Each line is of type "path speaker", as follows:

    path/to/1st.wav spk1
    path/to/2nd.wav spk2
    path/to/3rd.wav spk1

    and so on.
    min_mix and max_mix are the minimum and maximum number of examples to
    be mixed for generating a training example
    """
    speaker_wavs = {}
    noise_wavs = []
    batch_x = []
    batch_y = []
    batch_s = []
    batch_count = 0

    while True:  # Generate examples indefinitely
        # Select number of files to mix
        k = np.random.randint(min_mix, max_mix+1)
        if k > len(speaker_wavs):
            # Reading wav files list and separating per speaker
            speaker_wavs = {}
            f = open(wavlist)
            for line in f:
                line = line.strip().split()
                if len(line) != 2:
                    continue
                p, spk = line
                if spk not in speaker_wavs:
                    speaker_wavs[spk] = []
                speaker_wavs[spk].append(p)
            f.close()
            # Randomizing wav lists
            for spk in speaker_wavs:
                random.shuffle(speaker_wavs[spk])
        wavsum = None
        sigs = []
        ampsum = 0.

        # Pop wav files from random speakers, store them individually for
        # dominant spectra decision and generate the mixed input
        for spk in random.sample(speaker_wavs.keys(), k):
            p = speaker_wavs[spk].pop()
            if not speaker_wavs[spk]:
                del(speaker_wavs[spk])  # Remove empty speakers from dictionary
            sig, rate = sf.read(p)
            if rate != FRAME_RATE:
                raise Exception("Config specifies " + str(FRAME_RATE) +
                                "Hz as sample rate, but file " + str(p) +
                                "is in " + str(rate) + "Hz.")
            sig = sig - np.mean(sig)
            sig = sig/np.max(np.abs(sig))
            r = np.random.random()
            sig *= r
            ampsum += r
            if wavsum is None:
                wavsum = sig
            else:
                wavlen = min(len(wavsum), len(sig))
                beg1 = random.randint(0, len(wavsum) - wavlen)
                beg2 = random.randint(0, len(sig) - wavlen)
                sig = sig[beg2:wavlen + beg2]
                wavsum = wavsum[beg1:wavlen + beg1] + sig
                for i in range(len(sigs)):
                    sigs[i] = sigs[i][beg1:wavlen + beg1]
            sigs.append(sig)

        # Adding noise with chance of 50% if list is provided
        if noiselist is not None and random.random() < .5:
            # Repopulate noise wav list
            if len(noise_wavs) == 0:
                f = open(noiselist)
                for l in f:
                    l = l.strip()
                    if len(l) > 0:
                        noise_wavs.append(l)
                f.close()
                random.shuffle(noise_wavs)
            p = noise_wavs.pop()
            sig, rate = sf.read(p)
            if rate != FRAME_RATE:
                raise Exception("Config specifies " + str(FRAME_RATE) +
                                "Hz as sample rate, but file " + str(p) +
                                "is in " + str(rate) + "Hz.")
            wavlen = min(len(wavsum), len(sig))
            beg1 = random.randint(0, len(wavsum) - wavlen)
            beg2 = random.randint(0, len(sig) - wavlen)
            sig = sig[beg2:wavlen + beg2]
            sig = sig - np.mean(sig)
            sig = sig/np.max(np.abs(sig))
            r = np.random.random()
            sig *= r
            ampsum += r
            sig *= r
            wavsum = wavsum[beg1:wavlen + beg1] + sig
            for i in range(len(sigs)):
                sigs[i] = sigs[i][beg1:wavlen + beg1]
            sigs.append(sig)
        for i in range(len(sigs)):
            sig[i] /= ampsum
        wavsum /= ampsum

        # STFT for mixed signal
        X = np.real(np.log10(stft(wavsum) + 1e-7))
        if len(X) <= TIMESTEPS:
            continue

        # STFTs for individual signals
        specs = []
        for sig in sigs:
            specs.append(np.real(np.log10(stft(sig) + 1e-7)))
        specs = np.array(specs)

        nc = max_mix
        if noiselist is not None:
            nc += 1

        # Get dominant spectra indexes, create one-hot outputs
        Y = np.zeros(X.shape + (nc,))
        vals = np.argmax(specs, axis=0)
        for i in range(k):
            t = np.zeros(nc)
            t[i] = 1
            Y[vals == i] = t

        # Create mask for zeroing out gradients from silence components
        m = np.max(X) - 40./20  # Minus 40dB
        z = np.zeros(nc)
        Y[X < m] = z

        # EXPERIMENTAL: normalize log spectra as weighted norm vectors instead
        # of using unit vectors for "hard" classes
        S = np.zeros(X.shape + (nc,))
        S[:, :, :len(specs)] = np.transpose(specs, (1, 2, 0))
        S /= np.linalg.norm(S, axis=2, keepdims=True)

        # Generating sequences
        i = 0
        while i + TIMESTEPS < len(X):
            batch_x.append(X[i:i+TIMESTEPS])
            batch_y.append(Y[i:i+TIMESTEPS])
            batch_s.append(S[i:i+TIMESTEPS])
            i += TIMESTEPS//2

            batch_count = batch_count+1

            if batch_count == batch_size:
                inp = np.array(batch_x).reshape((batch_size,
                                                 TIMESTEPS, -1))
                hard_out = np.array(batch_y).reshape((batch_size,
                                                      TIMESTEPS, -1))
                soft_out = np.array(batch_s).reshape((batch_size,
                                                      TIMESTEPS, -1))
                yield({'input': inp},
                      {'hard_output': hard_out,
                       'soft_output': soft_out})
                batch_x = []
                batch_y = []
                batch_s = []
                batch_count = 0


if __name__ == "__main__":
    a = get_egs('train', 2, 4, 1)
    k = 6
    for i, j in a:
        print(i['input'].shape,
              j['hard_output'].shape,
              j['soft_output'].shape)
        print(j[0][0])
        k -= 1
        if k == 0:
            break
