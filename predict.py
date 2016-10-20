# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:20:31 2016

@author: valterf
STFT/ISTFT derived from Basj's implementation[1], with minor modifications,
such as the replacement of the hann window by its root square, as specified in
the original paper from Hershey et. al. (2015)[2]

[1] http://stackoverflow.com/a/20409020
[2] https://arxiv.org/abs/1508.04306
"""
import soundfile as sf
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from config import FRAME_LENGTH, FRAME_SHIFT


def sqrt_hann(M):
    return np.sqrt(np.hanning(M))


def stft(x, fftsize=int(FRAME_LENGTH*8000),
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


def prepare_features(wavpath, nnet, pred_index=1):
    freq = int(nnet.input.get_shape()[2])
    if(isinstance(nnet.output, list)):
        K = int(nnet.output[pred_index].get_shape()[2]) // freq
    else:
        K = int(nnet.output.get_shape()[2]) // freq
    sig, rate = sf.read(wavpath)
    if rate != 8000:
        raise Exception("Currently only 8000 Hz audio is supported. " +
                        "You have provided a {r} Hz one".format(r=rate))
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
    v = V.reshape((-1, K))

    return spec, rate, x, v


def soft_print_predict(wavpath, nnet, num_sources, wav_out=None, mask_power=3):
    k = num_sources
    freq = int(nnet.input.get_shape()[2])
    spec, rate, x, v = prepare_features(wavpath, nnet)

    imgs = []
    if k > 1:
        from sklearn.cluster import KMeans
        km = KMeans(k)
        km.fit(v)
        cc = km.cluster_centers_
    else:
        cc = [np.mean(v, axis=0)]

    for i in range(len(cc)):
        imgs.append(np.zeros((x.size)))
    for i in range(len(cc)):
        for j in range(len(imgs[i])):
            imgs[i][j] = euclidean(v[j], cc[i])
    imgs = np.array(imgs)
    for i in range(len(cc)):
        imgs[:, i] = np.max(imgs[:, i]) - imgs[:, i]
        imgs[:, i] /= np.linalg.norm(imgs[:, i], axis=-1, keepdims=True)
    for i in range(len(imgs)):
        imgs[i] /= np.max(imgs[i], axis=-1, keepdims=True)
    imgs **= mask_power

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(k+1, 1)
    for axis, img in zip(axes[:-1], imgs):
        mask = img.reshape(-1, freq)
        mask[:, [0, 1, -5, -4, -3, -2, -1]] /= 10
        mask[mask > 1] = 1
        mask /= np.max(mask)
        axis.imshow(mask.swapaxes(0, 1),
                    origin='lower', cmap='afmhot',
                    vmin=0, vmax=1)
    x -= np.min(x)
    x /= np.max(x)
    x **= 2
    axes[-1].imshow(x.reshape(-1, freq).swapaxes(0, 1),
                    origin='lower', cmap='afmhot')
    plt.show()
    if wav_out is None:
        return

    spec = np.log(spec)
    mag = np.real(spec)
    phase = np.imag(spec)
    i = 1
    for img in imgs:
        mask = img.reshape(-1, freq)
        mask[:, [0, -2, -1]] /= 10
        mask[mask > 1] = 1
        sig_out = istft(np.exp(mag + 1j * phase) * mask)
        sig_out -= np.mean(sig_out)
        sig_out /= np.max(sig_out)
        sf.write(wav_out + '{i}.wav'.format(i=i), sig_out, rate)
        i += 1


def hard_print_predict(wavpath, nnet, num_sources, wav_out=None):
    k = num_sources
    freq = int(nnet.input.get_shape()[2])
    spec, rate, x, v = prepare_features(wavpath, nnet, 1)

    from sklearn.cluster import KMeans
    km = KMeans(k)
    eg = km.fit_predict(v)

    imgs = np.zeros((k, eg.size))
    for i in range(k):
        imgs[i, eg == i] = 1

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(k+1, 1)
    for axis, img in zip(axes[:-1], imgs):
        axis.imshow(img.reshape(-1, freq).swapaxes(0, 1),
                    origin='lower', cmap='afmhot',
                    vmin=0, vmax=1)
    x -= np.min(x)
    x /= np.max(x)
    x **= 2
    axes[-1].imshow(x.reshape(-1, freq).swapaxes(0, 1),
                    origin='lower', cmap='afmhot')
    plt.show()
    if wav_out is None:
        return

    spec = np.log(spec)
    mag = np.real(spec)
    phase = np.imag(spec)
    i = 1
    for img in imgs:
        mask = img.reshape(-1, freq)
        sig_out = istft(np.exp(mag + 1j * phase) * mask)
        sig_out -= np.mean(sig_out)
        sig_out /= np.max(sig_out)
        sf.write(wav_out + '{i}.wav'.format(i=i), sig_out, rate)
        i += 1


def print_pca(wavpath, nnet, num_dims, wav_out=None):
    k = num_dims
    freq = int(nnet.input.get_shape()[2])
    spec, rate, x, v = prepare_features(wavpath, nnet, 1)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=k)
    egs = np.random.randint(len(v), size=2000)
    pca.fit(v[egs])
    imgs = pca.transform(v).swapaxes(0, 1)
    
    def get_mask(img):
        mask = img.reshape(-1, freq)
        mask += 0.02
        mask[mask > 1] = 1
        return mask

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(k+1, 1)
    for axis, img in zip(axes[:-1], imgs):
        mask = get_mask(img)
        axis.imshow(mask.swapaxes(0, 1),
                    origin='lower', cmap='afmhot',
                    vmin=-1, vmax=1)
    x -= np.min(x)
    x /= np.max(x)
    x **= 2
    axes[-1].imshow(x.reshape(-1, freq).swapaxes(0, 1),
                    origin='lower', cmap='afmhot')
    plt.show()
    if wav_out is None:
        return

    spec = np.log(spec)
    mag = np.real(spec)
    phase = np.imag(spec)
    i = 1
    for img in imgs:
        mask = get_mask(img)
        sig_out = istft(np.exp(mag + 1j * phase) * mask)
        sig_out -= np.mean(sig_out)
        sig_out /= np.max(sig_out)
        sf.write(wav_out + '{i}.wav'.format(i=i), sig_out, rate)
        i += 1


def print_lda(wavpath, nnet, num_dims, wav_out=None):
    k = num_dims
    freq = int(nnet.input.get_shape()[2])
    spec, rate, x, v = prepare_features(wavpath, nnet, 1)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.cluster import KMeans
    km = KMeans(k)
    cls = km.fit_predict(v)
    lda = LDA(n_components=k, solver='svd')
    egs = np.random.randint(len(v), size=2000)
    lda.fit(v[egs], cls[egs])
    imgs = lda.predict_proba(v).swapaxes(0, 1)

    def get_mask(img):
        mask = img.reshape(-1, freq)
        mask += .02
        mask[mask > 1] = 1
        return mask

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(k+1, 1)
    for axis, img in zip(axes[:-1], imgs):
        mask = get_mask(img)
        axis.imshow(mask.swapaxes(0, 1),
                    origin='lower', cmap='afmhot',
                    vmin=0, vmax=1)
    x -= np.min(x)
    x /= np.max(x)
    x **= 2
    axes[-1].imshow(x.reshape(-1, freq).swapaxes(0, 1),
                    origin='lower', cmap='afmhot')
    plt.show()

    if wav_out is None:
        return

    spec = np.log(spec)
    mag = np.real(spec)
    phase = np.imag(spec)
    i = 1
    for img in imgs:
        mask = get_mask(img)
        sig_out = istft(np.exp(mag + 1j * phase) * mask)
        sig_out -= np.mean(sig_out)
        sig_out /= np.max(sig_out)
        sf.write(wav_out + '{i}.wav'.format(i=i), sig_out, rate)
        i += 1
