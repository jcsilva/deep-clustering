# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:48:14 2016

@author: valterf
"""


def print_examples(x, y, v, num_classes, embedding_size,
                   mask=None,
                   soft_clustering=False,
                   show_subspace=True):
    from sklearn.cluster import KMeans
    from itertools import permutations
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import normalize
    from scipy.spatial.distance import cosine

    x = x[0][::2]
    y = y[0][::2]
    v = v[0][::2]
    v = normalize(v, axis=1)
    if mask is not None:
        mask = mask[0][::2]

    v = normalize(v, axis=1)
    k = num_classes
    K = embedding_size

    x = x.reshape((-1, 129))
    y = y.reshape((-1, 129, k))
    v = v.reshape((-1, K))
    v = normalize(v, axis=1)
    print(v[23])
    if mask is not None:
        mask = mask.reshape((-1,))
#        v[mask] = 0
        mask = mask.reshape((-1, 129))
        p = k + 1
    else:
        p = k

    model = KMeans(p)
    eg = model.fit_predict(v)
    imshape = x.shape + (3,)
    img = np.zeros((x.size, 3))

    # Hard clustering
    if not soft_clustering:
        img[eg == 0] = [1, 0, 0]
        img[eg == 1] = [0, 1, 0]
        if(p > 2):
            img[eg == 2] = [0, 0, 1]
            img[eg == 3] = [0, 0, 0]
        img = img.reshape(imshape)
        img2 = np.zeros(eg.shape + (3,))
        vals = np.argmax(y.reshape((-1, k)), axis=1)
        print(img2.shape, vals.shape)
        for i in range(k):
            t = np.zeros(3)
            t[i] = 1
            img2[vals == i] = t
        img2 = img2.reshape(imshape)
        if mask is not None:
            img2[mask] = [0, 0, 1]
    # Soft clustering
    else:
        cc = model.cluster_centers_
        for i in range(len(cc)):
            for j in range(len(img)):
                img[j][i] = (np.pi + cosine(v[j], cc[i])) / 2 / np.pi
            img[j] = np.max(img[j]) - img[j]
            img[j] = normalize(img[j])
        img = img.reshape(imshape)
        img2 = np.zeros(eg.shape + (3,))
        vals = y.reshape((-1, k))
        vals /= np.linalg.norm(vals, axis=-1, keepdims=True)
        for i in range(k):
            img2[:, i] = vals[:, i]
        img2 = img2.reshape(imshape)
        if mask is not None:
            img2[mask] = [0, 0, 1]

    # Log spectrum plot with better contrast
    img3 = x
    img3 -= np.min(img3)
    img3 **= 3

    # Find most probable color permutation from prediction
    p = None
    s = np.float('Inf')
    for pp in permutations([0, 1, 2]):
        ss = np.sum(np.square(img2 - img[:, :, pp]))
        if ss < s:
            s = ss
            p = pp
    img = img[:, :, p]
    img = img.reshape(imshape)
    img4 = 1 - (((img-img2+1)/2))
    img4[np.all(img4 == .5, axis=2)] = 0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.imshow(img.swapaxes(0, 1), origin='lower', cmap='afmhot')
    ax2.imshow(img2.swapaxes(0, 1), origin='lower', cmap='afmhot')
    ax3.imshow(img4.swapaxes(0, 1), origin='lower')
    ax4.imshow(img3.swapaxes(0, 1), origin='lower', cmap='afmhot')

    if show_subspace:
        import matplotlib.pyplot as plt
        #from sklearn.manifold import Isomap as manifold
        from sklearn.decomposition import PCA as manifold
        man = manifold(n_components=2)
        cls = eg
        egs = np.random.randint(len(v), size=500)
        k = man.fit_transform(v[egs, :])
        x, y = zip(*k)
        cls = cls[egs]
        plt.figure()
        for x, y, c in zip(x, y, cls):
            color = np.zeros(3)
            color[p[c]] = 1
            plt.scatter(x, y, color=color)
        plt.show()
