# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Input, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from keras.models import model_from_json
from feats import myGenerator, get_batches

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

EMBEDDINGS_DIMENSION = 50
NUM_CLASSES = 3
INPUT_SAMPLE_SIZE=12900

def save_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
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
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm
    
    # V e Y estao vetorizados
    # Antes de mais nada, volto ao formato de matrizes
    V = tf.reshape(V, [-1, EMBEDDINGS_DIMENSION])
    Y = tf.reshape(Y, [-1, NUM_CLASSES])
   
    T = tf.transpose
    dot = tf.matmul
    return norm(dot(T(V), V)) - 2 * norm(dot(T(V), Y)) + norm(dot(T(Y), Y))


def train_nnet():       
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
    model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=(100,129)))
    #model.add(Dropout(0.5))
    #model.add(GaussianNoise(0.77))    
    model.add(Bidirectional(LSTM(120)))    
    #model.add(GaussianNoise(0.77))
    #model.add(Dropout(0.5))
    model.add(Dense(INPUT_SAMPLE_SIZE * EMBEDDINGS_DIMENSION, init='uniform', activation='tanh'))
    #model.add(TimeDistributed(Dense(EMBEDDINGS_DIMENSION)))
    #model.add(Activation('softmax'))

    sgd = SGD(lr=1e-5, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss=affinitykmeans, optimizer=sgd)

    model.fit_generator(myGenerator(),  samples_per_epoch=50, nb_epoch=10, max_q_size=100)
    #score = model.evaluate(X_test, y_test, batch_size=16)
    save_model(model, "model")
    
def main():
    train_nnet()    
    loaded_model = load_model("model")

    xval = []
    yref = []
    ypred = []
    i = 0
    for x, y in myGenerator():
        i += 1
        #if i % 2 == 1:
        v = loaded_model.predict(x)
        xval.append(x.reshape(100, 129)[:50])
        yref.append(y.reshape(100, 129, NUM_CLASSES)[:50])
        ypred.append(v.reshape(100, 129, EMBEDDINGS_DIMENSION)[:50])
        if i == 6:
            break
#    #with open("outfile", "w") as f:
    xval = np.concatenate(xval)
    yref = np.concatenate(yref)
    ypred = np.concatenate(ypred)

    k = NUM_CLASSES
    model = KMeans(k)
    eg = model.fit_predict(ypred.reshape(ypred.size//50, 50))
    imshape = yref.shape
    img = np.zeros(eg.shape + (3,))
    img[eg == 0] = [1, 0, 0]
    img[eg == 1] = [0, 1, 0]
    if(k > 2):
        img[eg == 2] = [0, 0, 1]
        img[eg == 3] = [0, 0, 0]
    img = img.reshape(imshape)

    img2 = yref
    img3 = xval

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.imshow(img.swapaxes(0, 1), origin='lower')
    ax2.imshow(img2.swapaxes(0, 1), origin='lower')
    ax3.imshow(img3.swapaxes(0, 1), origin='lower')

if __name__ == "__main__":
    main()