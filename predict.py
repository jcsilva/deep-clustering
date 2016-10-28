# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:20:31 2016

@author: valterf
"""
import soundfile as sf
import numpy as np
from feats import stft, istft
from scipy.spatial.distance import cosine


def prepare_features(wavpath, nnet, pred_index=1):
    freq = int(nnet.input.get_shape()[2])
    if(isinstance(nnet.output, list)):
        K = int(nnet.output[pred_index].get_shape()[2]) // freq
    else:
        K = int(nnet.output.get_shape()[-1]) // freq
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
    print(V.shape, K)
    v = V.reshape((-1, 40))

    return spec, rate, x, v


def soft_print_predict(wavpath, nnet, num_sources, wav_out=None, mask_power=3):
    k = num_sources
    freq = int(nnet.input.get_shape()[2])
    spec, rate, x, v = prepare_features(wavpath, nnet)

    imgs = []
    if k > 1:
        from sklearn.cluster import MiniBatchKMeans as KMeans
        km = KMeans(k)
        km.fit(v)
        cc = km.cluster_centers_
    else:
        c = np.mean(v, axis=0)
        cc = [c / np.linalg.norm(c)]

    for i in range(len(cc)):
        imgs.append(np.zeros((x.size)))
    for i in range(len(cc)):
        for j in range(len(imgs[i])):
            imgs[i][j] = cosine(v[j], cc[i])
    imgs = np.array(imgs)
    for i in range(len(cc)):
        imgs[:, i] = np.max(imgs[:, i]) - imgs[:, i]
        if len(cc) > 1:
            imgs[:, i] /= np.linalg.norm(imgs[:, i], axis=-1, keepdims=True)
    for i in range(len(imgs)):
        imgs[i] /= np.max(imgs[i], axis=-1, keepdims=True)

    def get_mask(img):
        mask = img.reshape(-1, freq) ** mask_power
        mask -= np.mean(mask, axis=0, keepdims=True)
        mask /= np.std(mask, axis=0, keepdims=True)
        mask[mask > .5] = np.max(mask)
        return mask

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(k+1, 1)
    for axis, img in zip(axes[:-1], imgs):
        mask = get_mask(img)
        axis.imshow(mask.swapaxes(0, 1),
                    origin='lower', cmap='afmhot')
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
        sig_out[:freq//2] = 0
        sig_out[-freq//2:] = 0
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
        sig_out[:freq] = 0
        sig_out[-freq:] = 0
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
    lda = LDA(n_components=k, solver='eigen')
    egs = np.random.randint(len(v), size=2000)
    lda.fit(v[egs], cls[egs])
    imgs = lda.predict_proba(v).swapaxes(0, 1)

    def get_mask(img):
        mask = img.reshape(-1, freq)
        mask -= np.mean(mask, axis=0, keepdims=True)
        mask /= np.std(mask, axis=0, keepdims=True)
        mask[mask > 1] = np.max(mask)
        return mask

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(k+1, 1)
    for axis, img in zip(axes[:-1], imgs):
        mask = get_mask(img)
        axis.imshow(mask.swapaxes(0, 1),
                    origin='lower', cmap='afmhot')
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
