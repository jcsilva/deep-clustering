# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import sigproc
from keras.utils.np_utils import to_categorical
import random


FRAME_LENGTH = .032
FRAME_SHIFT = .008
FS = 8000 # Frequency sampling (Hz)

NUM_FRAMES = 100
CONTEXT = 100
CHUNK_SIZE = int((FRAME_LENGTH + (NUM_FRAMES-1) * FRAME_SHIFT) * FS)

def squared_hann(M):
    return np.sqrt(np.hanning(M))


def stft(sig, rate):
    frames = sigproc.framesig(sig,
                              FRAME_LENGTH*rate,
                              FRAME_SHIFT*rate,
                              winfunc=squared_hann)
    spec = np.fft.rfft(frames, int(FRAME_LENGTH*rate))
    return np.log(np.absolute(spec))


#def get_batches(batch_size):
#    p1 = "LapsBM_0061.wav"
#    rate, sig = wav.read(p1)
#    sig = sig - np.mean(sig)
#    sig = sig/np.max(np.abs(sig))
#
#    p2 = "LapsBM_0304.wav"
#    rate2, sig2 = wav.read(p2)
#    sig2 = sig2 - np.mean(sig2)
#    sig2 = sig2/np.max(np.abs(sig2))
#
#    sig = sig[len(sig2)-6592:len(sig2)]
#    sig2 = sig2[len(sig)-6592:len(sig)]
#    x = sig + sig2
#    
#    SPEC = stft(sig, rate)
#    SPEC2 = stft(sig2, rate)
#    X = stft(x, rate)
#    m = np.max(X) - 5.5 # consertar esse valor. Isso eh soh para identificar silencio!!
#    y = np.zeros(X.shape, dtype=np.int16)
#    y[SPEC > SPEC2] = 0
#    y[SPEC2 >= SPEC] = 1
#    y[X < m] = 2
#    
#    feats  = np.zeros((batch_size,) + X.shape)
#    labels = np.zeros((batch_size,) + y.shape, dtype=np.int16)
#    i = 0
#    for k in range(1):
#        feats[i] = X
#        labels[i] = y
#        i += 1
#        if i == batch_size:
#            i = 0
#            yield(feats.reshape((batch_size, X.size)),
#                  labels.reshape((batch_size, y.size)))
#            feats = np.zeros((batch_size,) + X.shape)
#            labels = np.zeros((batch_size,) + y.shape)

def get_signals():
    root_prefix = "/media/dados/resources/database/speech/pt-BR/LapsBM1.4-8k/all/LapsBM_"   
   
    file_num = str(np.random.randint(1,700))
    #file_num = "61"
    p1 = root_prefix + file_num.rjust(4,'0') + ".wav"
    rate, sig = wav.read(p1)
    sig = sig - np.mean(sig)
    sig = sig/np.max(np.abs(sig))

    #root_prefix = "/media/dados/resources/database/speech/pt-BR/LapsBM1.4-8k/LapsBM-M001/LapsBM_" 
    
    file_num = str(np.random.randint(1,700))
    #file_num = "1"
    p2 = root_prefix + file_num.rjust(4,'0') + ".wav"
    rate2, sig2 = wav.read(p2)
    sig2 = sig2 - np.mean(sig2)
    sig2 = sig2/np.max(np.abs(sig2))
    
    smallest_signal_len = len(sig) if len(sig) < len(sig2) else len(sig2)
    
    x = sig[:smallest_signal_len] + sig2[:smallest_signal_len]

    S1 = stft(sig[:smallest_signal_len], rate)
    S2 = stft(sig2[:smallest_signal_len], rate2)
    X = stft(x, rate) #[TODO] vou tratar de caso com taxa de amostragem diferente?
    
    return S1, S2, X

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
    
    
def myGenerator(n=0):
    counter = 0 
    
    def condition(counter, n):
        if n > 0:
            return counter < n
        else:
            return True
            
    while condition(counter, n):
        counter = counter + 1
        # escolho um arquivo
        S1, S2, X = get_signals()
        
        # cria mascara binaria
        m = np.max(X) - 5.5 # consertar esse valor. Isso eh soh para identificar silencio!!
        y = np.zeros(X.shape, dtype=np.int16)
        y[S1 > S2] = 0
        y[S2 >= S1] = 1
        y[X < m] = 2
        n_classes = np.max(y) + 1
        
        total_frames = X.shape[0]
        idx = 0
        while (idx + CONTEXT) < total_frames:
            
            #if i%10==0:
            #    print("i = " + str(i))
            
            # mlp
            #yield (X.reshape((1,-1)),
            #      to_categorical(y, NUM_CLASSES).reshape((1,-1)))
            
            # blstm
            yield (np.expand_dims(X[idx:idx + CONTEXT], axis=0),
                   to_categorical(np.ravel(y[idx:idx + CONTEXT]), n_classes).reshape((1,-1)))
            idx = idx + CONTEXT // 2


if __name__ == "__main__":
    a = myGenerator()
    for i,j in a:
        print(i.shape,j.shape)