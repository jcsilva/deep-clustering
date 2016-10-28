# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:09:46 2016

@author: valterf
"""
from nnet import train_nnet, load_model
from visualization import print_examples


def main():
    train_nnet('wavlist_short', 'wavlist_short')
    model = load_model('model')
    print_examples(["/media/misc/laps8k/LapsBM-F012/LapsBM_0221.wav",
                    "/media/misc/laps8k/LapsBM-M007/LapsBM_0124.wav"],
                   model)

if __name__ == "__main__":
    main()
