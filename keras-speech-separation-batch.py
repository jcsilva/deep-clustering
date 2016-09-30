# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Activation, Input, Reshape, BatchNormalization
from keras.layers import Dense, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from feats import myGenerator

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

EMBEDDINGS_DIMENSION = 50
NUM_CLASSES = 3


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


def get_dims(generator, embedding_size):
    inp, out = next(generator)
    inp_shape = (None, inp.shape[-1])
    out_shape = (None, out.shape[-1]//NUM_CLASSES * embedding_size)
    return inp_shape, out_shape


def affinitykmeans(Y, V):
    def norm(tensor):
        square_tensor = K.square(tensor)
        frobenius_norm2 = K.sum(square_tensor)
        frobenius_norm = K.sqrt(frobenius_norm2)
        return frobenius_norm

    # V e Y estao vetorizados
    # Antes de mais nada, volto ao formato de matrizes
    V = K.l2_normalize(K.reshape(V, [-1, EMBEDDINGS_DIMENSION]), axis=1)
    Y = K.reshape(Y, [-1, NUM_CLASSES])

    T = K.transpose
    dot = K.dot
    return norm(dot(T(V), V)) - 2 * norm(dot(T(V), Y)) + norm(dot(T(Y), Y))


def train_nnet():
    inp_shape, out_shape = get_dims(myGenerator(),
                                    EMBEDDINGS_DIMENSION)
#    model = Sequential()
#    model.add(Dense(64, input_dim=INPUT_SAMPLE_SIZE, init='uniform'))
#    model.add(Activation('tanh'))
#    model.add(Dropout(0.5))
#    model.add(Dense(64, init='uniform'))
#    model.add(Activation('tanh'))
#    model.add(Dropout(0.5))
#    model.add(Dense(INPUT_SAMPLE_SIZE * EMBEDDINGS_DIMENSION, init='uniform'))
#    model.add(Activation('softmax'))

    model = Sequential()
    model.add(Bidirectional(LSTM(600, return_sequences=True),
                            input_shape=inp_shape))
    model.add(TimeDistributed(BatchNormalization(mode=2)))
    model.add(GaussianNoise(0.77))
    # model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(600, return_sequences=True)))
    model.add(TimeDistributed(BatchNormalization(mode=2)))
    model.add(GaussianNoise(0.77))
    # model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(out_shape[-1],
                                    init='uniform',
                                    activation='tanh')))

    sgd = RMSprop()

    model.compile(loss=affinitykmeans, optimizer=sgd)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5)

    checkpoint = ModelCheckpoint('weights-{epoch:03d}-{val_loss:.2f}.hdf5',
                                 save_weights_only=True)

    model.fit_generator(myGenerator(1, 600),
                        samples_per_epoch=800,
                        nb_epoch=200,
                        max_q_size=800,
                        validation_data=myGenerator(601, 700),
                        nb_val_samples=100,
                        callbacks=[earlyStopping, checkpoint])
    # score = model.evaluate(X_test, y_test, batch_size=16)
    save_model(model, "model")


def main():
    train_nnet()
    loaded_model = load_model("model")

    x, y = next(myGenerator(601, 700))
    v = loaded_model.predict(x)
    x = x[:][::2]
    y = y[:][::2]
    v = v[:][::2]
    x = x.reshape((-1, 129))
    y = y.reshape((-1, 129, NUM_CLASSES))
    v = v.reshape((-1, 129, EMBEDDINGS_DIMENSION))

    k = NUM_CLASSES
    model = KMeans(k)
    eg = model.fit_predict(v.reshape(-1, EMBEDDINGS_DIMENSION))
    imshape = y.shape
    img = np.zeros(eg.shape + (3,))
    img[eg == 0] = [1, 0, 0]
    img[eg == 1] = [0, 1, 0]
    if(k > 2):
        img[eg == 2] = [0, 0, 1]
        img[eg == 3] = [0, 0, 0]
    img = img.reshape(imshape)

    img2 = y
    img3 = x

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.imshow(img.swapaxes(0, 1), origin='lower')
    ax2.imshow(img2.swapaxes(0, 1), origin='lower')
    ax3.imshow(img3.swapaxes(0, 1), origin='lower')

if __name__ == "__main__":
    main()
