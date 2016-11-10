# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras.layers import TimeDistributed, Bidirectional
from keras.regularizers import l2
from keras.optimizers import Nadam
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from feats import get_egs

from config import NUM_RLAYERS, SIZE_RLAYERS
from config import BATCH_SIZE, SAMPLES_PER_EPOCH, NUM_EPOCHS, VALID_SIZE
from config import DROPOUT, RDROPOUT, L2R, CLIPNORM


def get_dims(generator):
    inp, out = next(generator)
    inp_shape = (None, inp['input'].shape[-1])
    return inp_shape, inp_shape


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


def train_nnet(train_list, valid_list, weights_path=None):
    train_gen = get_egs(train_list,
                        'noisel',
                        batch_size=BATCH_SIZE)
    valid_gen = get_egs(valid_list,
                        'noisel',
                        batch_size=BATCH_SIZE)
    inp_shape, out_shape = get_dims(train_gen)

    inp = Input(shape=inp_shape, name='input')
    x = inp
    for i in range(NUM_RLAYERS):
        x = Bidirectional(LSTM(SIZE_RLAYERS, return_sequences=True,
                               W_regularizer=l2(L2R),
                               U_regularizer=l2(L2R),
                               b_regularizer=l2(L2R),
                               dropout_W=DROPOUT,
                               dropout_U=RDROPOUT),
                          input_shape=inp_shape)(x)
    kmeans_o = TimeDistributed(Dense(out_shape[-1],
                                     activation='linear',
                                     W_regularizer=l2(L2R),
                                     b_regularizer=l2(L2R)),
                               name='irm')(x)

    model = Model(input=[inp], output=[kmeans_o])
    if weights_path:
        model.load_weights(weights_path)
    model.compile(loss={'irm': 'mse'},
                  optimizer=Nadam(clipnorm=CLIPNORM))

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
                        nb_val_samples=VALID_SIZE,
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=NUM_EPOCHS,
                        max_q_size=512,
                        callbacks=callbacks_list)
    save_model(model, "model")
