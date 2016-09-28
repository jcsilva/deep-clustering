# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
"""

import theano as th
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Input, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from keras.models import model_from_json
from feats import myGenerator

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
        square_tensor = th.tensor.sqr(tensor)
        frobenius_norm_sqr = th.tensor.sum(square_tensor)
        #frobenius_norm = th.tensor.sqrt(frobenius_norm_sqr)
        return frobenius_norm_sqr
    
    # V e Y estao vetorizados
    # Antes de mais nada, volto ao formato de matrizes
    V = th.tensor.reshape(V, [-1, EMBEDDINGS_DIMENSION])
    Y = th.tensor.reshape(Y, [-1, NUM_CLASSES])
   
    #T = th.transpose
    dot = th.dot
    return norm(dot(V.T, V)) - 2 * norm(dot(V.T, Y)) + norm(dot(Y.T, Y))


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
    model.add(Bidirectional(LSTM(300)))    
    #model.add(GaussianNoise(0.77))
    #model.add(Dropout(0.5))
    model.add(Dense(INPUT_SAMPLE_SIZE * EMBEDDINGS_DIMENSION, init='uniform', activation='tanh'))
    #model.add(TimeDistributed(Dense(EMBEDDINGS_DIMENSION)))
    #model.add(Activation('softmax'))

    sgd = SGD(lr=1e-5, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss=affinitykmeans, optimizer=sgd)

    model.fit_generator(myGenerator(),  samples_per_epoch=10, nb_epoch=1, max_q_size=10)
    #score = model.evaluate(X_test, y_test, batch_size=16)
    save_model(model, "model")
    
def main():
    train_nnet()    
    loaded_model = load_model("model")

    ypred = []
    for x,y in myGenerator(1):
        ypred.append(loaded_model.predict(x))
    #with open("outfile", "w") as f:
    for y in ypred:
        print(y.shape)


if __name__ == "__main__":
    main()