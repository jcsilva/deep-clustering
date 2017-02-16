# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:09:46 2016

@author: valterf

Main code with examples for the most important function calls. None of this
will work if you haven't prepared your train/valid/test file lists.
"""
from visualization import print_examples
from nnet import train_nnet, load_model
from predict import separate_sources


def main():
    train_nnet('train', 'valid')
    model = load_model('model')
    egs = []
    current_spk = ""

    # From here on, all the code does is get 2 random speakers from the test
    # set and visualize the outputs and references. You need to have matplotlib
    # installed for this to work.
    for line in open('test'):
        line = line.strip().split()
        if len(line) != 2:
            continue
        w, s = line
        if s != current_spk:
            egs.append(w)
            current_spk = s
            if len(egs) == 2:
                break
    print_examples(egs, model, db_threshold=40, ignore_background=True)
    
    # If you wish to test source separation, generate a mixed 'mixed.wav'
    # file and test with the following line
    # separate_sources('mixed.wav', model, 2, 'out')


if __name__ == "__main__":
    main()
