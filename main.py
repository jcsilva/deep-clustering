# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:09:46 2016

@author: valterf

Main code with examples for the most important function calls. None of this
will work if you haven't prepared your train/valid/test file lists.
"""
from visualization import print_examples
from nnet import train_nnet, load_model


def main():
    train_nnet('wavlist', 'wavlist')


if __name__ == "__main__":
    main()
