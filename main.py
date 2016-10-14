# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:09:46 2016

@author: valterf
"""
import numpy as np
from visualization import print_examples
from nnet import train_nnet, load_model, SIL_AS_CLASS, EMBEDDINGS_DIMENSION
from feats import get_egs


def main():
#    train_nnet('wavlist_short', 'wavlist_short')
    loaded_model = load_model("model")
    X = []
    Y = []
    V = []
    gen = get_egs('wavlist_short', 2, 2, SIL_AS_CLASS)
    i = 0
    for inp, ref in gen:
        inp, ref = next(gen)
        X.append(inp['input'])
        Y.append(ref['hard_output'])
        V.append(loaded_model.predict(inp)[1])
        i += 1
        if i == 8:
            break
    x = np.concatenate(X, axis=1)
    y = np.concatenate(Y, axis=1)
    v = np.concatenate(V, axis=1)

    np.save('x', x)
    np.save('y', y)
    np.save('v', v)

    x = np.load('x.npy')
    y = np.load('y.npy')
    v = np.load('v.npy')
    m = np.max(x) - 2
    print_examples(x, y, v, 2, EMBEDDINGS_DIMENSION)


if __name__ == "__main__":
    main()
