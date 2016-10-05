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
from keras.optimizers import Adam, SGD, Adadelta
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from feats import get_egs
import numpy as np

EMBEDDINGS_DIMENSION = 20
NUM_CLASSES = 2
SIL_AS_CLASS = False

BATCH_SIZE = 10
TIMESTEPS = 100
FREQSTEPS = 129


def print_examples(x, y, v):
    from sklearn.cluster import KMeans
    from itertools import permutations
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    x = x[0][::2]
    y = y[0][::2]
    v = v[0][::2]

    v = normalize(v, axis=1)
    k = NUM_CLASSES + int(SIL_AS_CLASS)

    x = x.reshape((-1, 129))
    y = y.reshape((-1, 129, k))
    v = v.reshape((-1, 129, EMBEDDINGS_DIMENSION))

    model = KMeans(k)
    eg = model.fit_predict(v.reshape(-1, EMBEDDINGS_DIMENSION))
    imshape = x.shape + (3,)
    img = np.zeros(eg.shape + (3,))
    img[eg == 0] = [1, 0, 0]
    img[eg == 1] = [0, 1, 0]
    if(k > 2):
        img[eg == 2] = [0, 0, 1]

    img = img.reshape(imshape)

    img2 = np.zeros(eg.shape + (3,))
    vals = np.argmax(y.reshape((-1, k)), axis=1)

    for i in range(k):
        t = np.zeros(3)
        t[i] = 1
        img2[vals == i] = t
    img2 = img2.reshape(imshape)

    img3 = x

    # Find most probable color permutation from prediction
  #  p = None
  #  s = np.float('Inf')
  #  for pp in permutations([0, 1, 2]):
  #      ss = np.sum(np.square(img2 - img[:, :, pp]))
  #      if ss < s:
  #          s = ss
  #          p = pp
  #  img = img[:, :, p]
  #  img4 = 1 - (((img-img2+1)/2))
  #  img4[np.all(img4 == .5, axis=2)] = 0

  #  fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
  #  ax1.imshow(img.swapaxes(0, 1), origin='lower')
  #  ax2.imshow(img2.swapaxes(0, 1), origin='lower')
  #  ax3.imshow(img4.swapaxes(0, 1), origin='lower')
  #  ax4.imshow(img3.swapaxes(0, 1), origin='lower')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.imshow(img.swapaxes(0, 1), origin='lower')
    ax2.imshow(img2.swapaxes(0, 1), origin='lower')
    ax3.imshow(img3.swapaxes(0, 1), origin='lower')


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
    print("Model saved to disk")


def load_model(filename):
    # load json and create model
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename + ".h5")
    print("Model loaded from disk")
    return loaded_model


def affinitykmeans(Y, V):
    def norm(tensor):
        square_tensor = K.square(tensor)
        tensor_sum = K.sum(square_tensor, axis=(1, 2))
        frobenius_norm = K.sqrt(tensor_sum)
        return frobenius_norm

    def dot(x, y):
        return K.batch_dot(x, y, axes=(2, 1))

    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])

    V = K.l2_normalize(K.reshape(V, [BATCH_SIZE,
                                     TIMESTEPS*FREQSTEPS,
                                     EMBEDDINGS_DIMENSION]), axis=-1)
    Y = K.reshape(Y, [BATCH_SIZE,
                      TIMESTEPS*FREQSTEPS,
                      NUM_CLASSES + int(SIL_AS_CLASS)])

    silence_mask = K.sum(Y, axis=2, keepdims=True)
    V = silence_mask * V

    return norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))


def train_nnet(train_list, valid_list, weights_path=None):
    train_gen = get_egs(train_list,
                        min_mix=NUM_CLASSES,
                        max_mix=NUM_CLASSES,
                        sil_as_class=SIL_AS_CLASS,
                        batch_size=BATCH_SIZE)
    valid_gen = get_egs(valid_list,
                        min_mix=NUM_CLASSES,
                        max_mix=NUM_CLASSES,
                        sil_as_class=SIL_AS_CLASS,
                        batch_size=BATCH_SIZE)
    inp_shape, out_shape = get_dims(train_gen,
                                    EMBEDDINGS_DIMENSION)
    model = Sequential()
    # model.add(BatchNormalization(mode=2, input_shape=inp_shape))
    model.add(Bidirectional(LSTM(120, return_sequences=True,
                                 activation='tanh'),
                            input_shape=inp_shape))
    # model.add(TimeDistributed(BatchNormalization(mode=2)))
    model.add(TimeDistributed((GaussianNoise(0.775))))
#    model.add(TimeDistributed((Dropout(0.5))))
    model.add(Bidirectional(LSTM(120, return_sequences=True,
                                 activation='tanh')))
    # model.add(TimeDistributed(BatchNormalization(mode=2)))
    model.add(TimeDistributed((GaussianNoise(0.775))))
#    model.add(TimeDistributed((Dropout(0.5))))
    model.add(TimeDistributed(Dense(out_shape[-1],
                                    init='uniform',
                                    activation='tanh')))

#    sgd = SGD(lr=1e-5, momentum=0.9, decay=0.0, nesterov=True)
    sgd = Adadelta()
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
                        nb_val_samples=1,
                        samples_per_epoch=200,
                        nb_epoch=50,
                        max_q_size=10,
                        callbacks=callbacks_list)
    # score = model.evaluate(X_test, y_test, batch_size=16)
    save_model(model, "model")


def main():
    train_nnet('wavlist_short', 'wavlist_short')
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

    print_examples(x, y, v)


if __name__ == "__main__":
    main()
