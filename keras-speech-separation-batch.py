# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Activation, Input, Reshape
from keras.layers import Dense, LSTM, BatchNormalization
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from feats import get_egs
import numpy as np

EMBEDDINGS_DIMENSION = 50
NUM_CLASSES = 2
SIL_AS_CLASS = False


def print_examples(x, y, v):
    from sklearn.cluster import KMeans
    from itertools import permutations
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    x = x[0][::2]
    y = y[0][::2]
    v = v[0][::2]

    v = normalize(v, axis=1)
    k = NUM_CLASSES
    if SIL_AS_CLASS:
        k += 1

    x = x.reshape((-1, 129))
    y = y.reshape((-1, 129, k))
    v = v.reshape((-1, 129, EMBEDDINGS_DIMENSION))

    model = KMeans(k+1)
    eg = model.fit_predict(v.reshape(-1, EMBEDDINGS_DIMENSION))
    imshape = x.shape + (3,)
    img = np.zeros(eg.shape + (3,))
    img[eg == 0] = [1, 0, 0]
    img[eg == 1] = [0, 1, 0]
    if(k > 2):
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

    img3 = x

    # Find most probable color permutation from prediction
    p = None
    s = np.float('Inf')
    for pp in permutations([0, 1, 2]):
        ss = np.sum(np.square(img2 - img[:, :, pp]))
        if ss < s:
            s = ss
            p = pp
    img = img[:, :, p]
    img4 = 1 - (((img-img2+1)/2))
    img4[np.all(img4 == .5, axis=2)] = 0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.imshow(img.swapaxes(0, 1), origin='lower')
    ax2.imshow(img2.swapaxes(0, 1), origin='lower')
    ax3.imshow(img4.swapaxes(0, 1), origin='lower')
    ax4.imshow(img3.swapaxes(0, 1), origin='lower')


def get_dims(generator, embedding_size):
    inp, out = next(generator)
    k = NUM_CLASSES + int(SIL_AS_CLASS)
    inp_shape = inp.shape[1:]
    out_shape = list(out.shape[1:])
    out_shape[-1] *= embedding_size/k
    out_shape[-1] = int(out_shape[-1])
    out_shape = tuple(out_shape)
    return inp_shape, out_shape


def save_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename + ".h5")
    print("Saved model to disk")


def load_model(filename):
    # load json and create model
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename + ".h5")
    print("Loaded model from disk")
    return loaded_model


def affinitykmeans(Y, V):
    def norm(tensor):
        square_tensor = K.square(tensor)
        tensor_sum = K.sum(square_tensor)
        frobenius_norm = K.sqrt(tensor_sum)
        return frobenius_norm

    # V e Y estao vetorizados
    # Antes de mais nada, volto ao formato de matrizes
    V = K.l2_normalize(K.reshape(V, [-1, EMBEDDINGS_DIMENSION]), axis=1)
    Y = K.reshape(Y, [-1, NUM_CLASSES + int(SIL_AS_CLASS)])

    T = K.transpose
    dot = K.dot
    return norm(dot(T(V), V)) - 2 * norm(dot(T(V), Y)) + norm(dot(T(Y), Y))


def train_nnet(train_list, valid_list, weights_path=None):
    train_gen = get_egs(train_list,
                        min_mix=NUM_CLASSES,
                        max_mix=NUM_CLASSES,
                        sil_as_class=SIL_AS_CLASS)
    valid_gen = get_egs(valid_list,
                        min_mix=NUM_CLASSES,
                        max_mix=NUM_CLASSES,
                        sil_as_class=SIL_AS_CLASS)
    inp_shape, out_shape = get_dims(train_gen,
                                    EMBEDDINGS_DIMENSION)
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True),
                            input_shape=inp_shape))
    model.add(TimeDistributed(BatchNormalization(mode=2)))
    model.add(TimeDistributed((GaussianNoise(0.775))))
#    model.add(TimeDistributed((Dropout(0.5))))
    model.add(Bidirectional(LSTM(30, return_sequences=True)))
    model.add(TimeDistributed(BatchNormalization(mode=2)))
    model.add(TimeDistributed((GaussianNoise(0.775))))
#    model.add(TimeDistributed((Dropout(0.5))))
    model.add(TimeDistributed(Dense(out_shape[-1],
                                    init='uniform',
                                    activation='relu')))

#    sgd = SGD(lr=1e-5, momentum=0.9, decay=0.0, nesterov=True)
    sgd = Adam()
    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss=affinitykmeans, optimizer=sgd)

    # checkpoint
    filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    callbacks_list = [checkpoint]

    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        nb_val_samples=128,
                        samples_per_epoch=65536,
                        nb_epoch=20,
                        max_q_size=10,
                        callbacks=callbacks_list)
    # score = model.evaluate(X_test, y_test, batch_size=16)
    save_model(model, "model")


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
        X.append(inp)
        Y.append(ref)
        V.append(loaded_model.predict(inp))
        i += 1
        if i == 8:
            break
    x = np.concatenate(X, axis=1)
    y = np.concatenate(Y, axis=1)
    v = np.concatenate(V, axis=1)

#    np.save('x', x)
#    np.save('y', y)
#    np.save('v', v)
#
#    x = np.load('x.npy')
#    y = np.load('y.npy')
#    v = np.load('v.npy')
    print_examples(x, y, v)


if __name__ == "__main__":
    main()