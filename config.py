# -*- coding: utf-8 -*-

# Audio mixing and FFT parameters
FRAME_RATE = 8000
FRAME_LENGTH = .032
FRAME_SHIFT = .008
TIMESTEPS = 100
DB_THRESHOLD = 40

# Neural net topology
SIZE_RLAYERS = 30
NUM_RLAYERS = 2

# Training parameters
BATCH_SIZE = 10
SAMPLES_PER_EPOCH = 100
NUM_EPOCHS = 50
VALID_SIZE = 10

# Regularization parameters
DROPOUT = 0.5              # Feed forward dropout
RDROPOUT = 0.2             # Recurrent dropout
L2R = 1e-6                 # L2 regularization factor
CLIPNORM = 200             # Norm clipping for gradients
