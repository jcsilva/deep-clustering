# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import sigproc
from keras.utils.np_utils import to_categorical

FRAME_LENGTH = .032
FRAME_SHIFT = .008
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


def get_signals(min_idx=1, max_idx=700):
    root_prefix = "/media/data/corpora/Laps/LapsBM1.4-8k/all/LapsBM_"   
    file_num = str(np.random.randint(min_idx, max_idx))
    #file_num = "61"
    p1 = root_prefix + file_num.rjust(4,'0') + ".wav"
    rate, sig = wav.read(p1)
    sig = sig - np.mean(sig)
    sig = sig/np.max(np.abs(sig))

    file_num = str(np.random.randint(min_idx, max_idx))
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
    
    
def myGenerator(min_idx=1, max_idx=700, n=0):
    counter = 0 
    
    def condition(counter, n):
        if n > 0:
            return counter < n
        else:
            return True
            
    while condition(counter, n):
        counter = counter + 1
        # escolho um arquivo
        S1, S2, X = get_signals(min_idx,max_idx)
        
        # cria mascara binaria
        m = np.max(X) - 5.5 # consertar esse valor. Isso eh soh para identificar silencio!!
        y = np.zeros(X.shape, dtype=np.int16)
        y[S1 > S2] = 0
        y[S2 >= S1] = 1
        y[X < m] = 2

        total_frames = X.shape[0]
        idx = 0
        total_x = []
        total_y = []
        while (idx + CONTEXT) < total_frames:
            # blstm
            # ACUMULAR ENTRADA E SAIDA DA FORMA: (1, n, 12900), (n, 12900*n_classes),
            # sendo n o numero de iteracoes por esse loop
            yield (np.expand_dims(X[idx:idx + CONTEXT], axis=0),
            np.expand_dims(to_categorical(np.ravel(y[idx:idx + CONTEXT])).reshape((100,-1)), axis=0))
            idx = idx + CONTEXT // 2
            #yield(np.array(total_x).transpose([1,0,2]), np.array(total_y).transpose([1,0,2]))

        
if __name__ == "__main__":
    a = myGenerator(1)
    for i,j in a:
        print(i.shape,j.shape)