# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""
import numpy as np
import math
import scipy.io.wavfile as wav
from python_speech_features import sigproc
from keras.utils.np_utils import to_categorical

FRAME_LENGTH = .032
FRAME_SHIFT = .008
FS = 8000 # Frequency sampling (Hz)

NUM_FRAMES = 100
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
    root_prefix = "/media/data/corpora/Laps/LapsBM1.4-8k/all/LapsBM_"   
   
    file_num = str(np.random.randint(1,700))
    p1 = root_prefix + file_num.rjust(4,'0') + ".wav"
    rate, sig = wav.read(p1)
    sig = sig - np.mean(sig)
    sig = sig/np.max(np.abs(sig))

    file_num = str(np.random.randint(1,700))
    p2 = root_prefix + file_num.rjust(4,'0') + ".wav"
    rate2, sig2 = wav.read(p2)
    sig2 = sig2 - np.mean(sig2)
    sig2 = sig2/np.max(np.abs(sig2))

    return sig, sig2    


def myGenerator(n=0):
    counter = 0 
    
    def condition(counter, n):
        if n > 0:
            return counter < n
        else:
            return True
            
    while condition(counter, n):
        counter = counter + 1
        s1, s2 = get_signals()
        smallest_signal_len = len(s1) if len(s1) < len(s2) else len(s2)
        total_frames = math.floor( (smallest_signal_len - int(FRAME_LENGTH * FS)) / int(FRAME_SHIFT * FS) ) + 1
        for i in range(0, total_frames, NUM_FRAMES // 2):
            # seleciona um frame
            sig  = s1[int(i*FRAME_SHIFT):int(i*FRAME_SHIFT) + CHUNK_SIZE]
            sig2 = s2[int(i*FRAME_SHIFT):int(i*FRAME_SHIFT) + CHUNK_SIZE]
            
            # cria mistura            
            x = sig + sig2
            
            # Espectro de magnitude
            SPEC = stft(sig, FS)
            SPEC2 = stft(sig2, FS)
            X = stft(x, FS)
            
            # cria mascara binaria
            m = np.max(X) - 5.5 # consertar esse valor. Isso eh soh para identificar silencio!!
            y = np.zeros(X.shape, dtype=np.int16)
            y[SPEC > SPEC2] = 0
            y[SPEC2 >= SPEC] = 1
            y[X < m] = 2
            y = np.squeeze(y.reshape((1,-1)))
            n_classes = np.max(y) + 1
            
            #if i%10==0:
            #    print("i = " + str(i))
            
            # mlp
            #yield (X.reshape((1,-1)),
            #      to_categorical(y, NUM_CLASSES).reshape((1,-1)))
            
            # blstm
            yield (X.reshape((1,X.shape[0], X.shape[1])),
                  to_categorical(y, n_classes).reshape((1,-1)))


if __name__ == "__main__":
    a = myGenerator(1)
    for i,j in a:
        print(i.shape,j.shape)