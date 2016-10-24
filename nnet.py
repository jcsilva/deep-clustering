# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, BatchNormalization
from keras.layers import TimeDistributed, Bidirectional
from keras.regularizers import l2
from keras.optimizers import Nadam
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from feats import get_egs, get_egs_end2end

from config import EMBEDDINGS_DIMENSION, MIN_MIX, MAX_MIX, TIMESTEPS
from config import NUM_RLAYERS, SIZE_RLAYERS
from config import BATCH_SIZE, SAMPLES_PER_EPOCH, NUM_EPOCHS, VALID_SIZE
from config import DROPOUT, RDROPOUT, L2R, CLIPNORM


def get_dims(generator, embedding_size):
    inp, out = next(generator)
    k = MAX_MIX
    inp_shape = (None, inp['input'].shape[-1])
    out_shape = list(out['kmeans_o'].shape[1:])
    out_shape[-1] *= float(embedding_size)/k
    out_shape[-1] = int(out_shape[-1])
    out_shape = tuple(out_shape)
    return inp_shape, out_shape


def get_dims_end2end(generator):
    inp, out = next(generator)
    inp_shape = (None, inp['input'].shape[-1])
    out_shape = (None, out['spec_o'].shape[-1])
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
        frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
        return frobenius_norm2

    def dot(x, y):
        return K.batch_dot(x, y, axes=(2, 1))

    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])

    V = K.reshape(V, [BATCH_SIZE, -1,
                      EMBEDDINGS_DIMENSION])
    Y = K.reshape(Y, [BATCH_SIZE, -1, MAX_MIX])

    return norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))


def get_kmeans_nnet(train_gen):
    inp_shape, out_shape = get_dims(train_gen,
                                    EMBEDDINGS_DIMENSION)
    inp = Input(shape=inp_shape, name='input')
    x = inp
    for i in range(NUM_RLAYERS):
        x = Bidirectional(LSTM(SIZE_RLAYERS, return_sequences=True,
                               W_regularizer=l2(L2R),
                               U_regularizer=l2(L2R),
                               b_regularizer=l2(L2R),
                               dropout_W=DROPOUT,
                               dropout_U=RDROPOUT))(x)
    kmeans_o = TimeDistributed(Dense(out_shape[-1],
                                     activation='tanh',
                                     W_regularizer=l2(L2R),
                                     b_regularizer=l2(L2R)),
                               name='kmeans_o')(x)
    model = Model(input=[inp], output=[kmeans_o])
    model.compile(loss={'kmeans_o': affinitykmeans},
                  optimizer=Nadam(clipnorm=CLIPNORM))
    return model


def get_enhancement_nnet(train_gen):
    inp_shape, out_shape = get_dims_end2end(train_gen)
    inp = Input(shape=inp_shape, name='input')
    x = inp
    for i in range(2):
        x = Bidirectional(LSTM(SIZE_RLAYERS, return_sequences=True,
                               W_regularizer=l2(L2R),
                               U_regularizer=l2(L2R),
                               b_regularizer=l2(L2R),
                               dropout_W=DROPOUT,
                               dropout_U=RDROPOUT))(x)
        x = BatchNormalization(mode=2)(x)
    spec_o = TimeDistributed(Dense(out_shape[-1],
                                   activation='linear',
                                   W_regularizer=l2(L2R),
                                   b_regularizer=l2(L2R)),
                             name='spec_o')(x)
    model = Model(input=[inp], output=[spec_o])
    model.compile(loss={'spec_o': 'mse'},
                  optimizer=Nadam(clipnorm=CLIPNORM))
    return model


def train_nnet(train_list, valid_list, weights_path=None):
    train_gen = get_egs(train_list,
                        min_mix=MIN_MIX,
                        max_mix=MAX_MIX,
                        batch_size=BATCH_SIZE)
    valid_gen = get_egs(valid_list,
                        min_mix=MIN_MIX,
                        max_mix=MAX_MIX,
                        batch_size=BATCH_SIZE)
    model = get_kmeans_nnet(train_gen)
    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        nb_val_samples=VALID_SIZE,
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=NUM_EPOCHS,
                        max_q_size=512)
    save_model(model, "model")
    model = load_model('model')
    train_gen = get_egs_end2end(train_list, model,
                                min_mix=MIN_MIX,
                                max_mix=MAX_MIX,
                                batch_size=BATCH_SIZE)
    valid_gen = get_egs_end2end(valid_list, model,
                                min_mix=MIN_MIX,
                                max_mix=MAX_MIX,
                                batch_size=BATCH_SIZE)
    model2 = get_enhancement_nnet(train_gen)
    model2.fit_generator(train_gen,
                         validation_data=valid_gen,
                         nb_val_samples=VALID_SIZE,
                         samples_per_epoch=SAMPLES_PER_EPOCH,
                         nb_epoch=NUM_EPOCHS,
                         max_q_size=512)
    save_model(model2, "model2")
