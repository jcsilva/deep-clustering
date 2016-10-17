# -*- coding: utf-8 -*-

# FFT parameters
FRAME_LENGTH = .032
FRAME_SHIFT = .008
TIMESTEPS = 100

# Clustering parameters
EMBEDDINGS_DIMENSION = 20
MIN_MIX = 2
MAX_MIX = 2
SIL_AS_CLASS = False

# Neural net topology
SIZE_RLAYERS = 300
NUM_RLAYERS = 2

# Training parameters
BATCH_SIZE = 1
SAMPLES_PER_EPOCH = 300
NUM_EPOCHS = 1
VALID_SIZE = 10

# Regularization parameters
DROPOUT = 0.5   # Feed forward dropout
RDROPOUT = 0.2  # Recurrent dropout
L2R = 1e-3      # L2 regularization factor
CLIPNORM = 200  # Norm clipping for gradients
