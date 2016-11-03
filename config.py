# -*- coding: utf-8 -*-

# Audio mixing and FFT parameters
FRAME_RATE = 8000
FRAME_LENGTH = .032
FRAME_SHIFT = .008
TIMESTEPS = 100
DB_THRESHOLD = 40

# Clustering parameters
EMBEDDINGS_DIMENSION = 40
MIN_MIX = 2
MAX_MIX = 3

# Neural net topology
SIZE_RLAYERS = 300
NUM_RLAYERS = 2

# Training parameters
BATCH_SIZE = 128
SAMPLES_PER_EPOCH = 8192
NUM_EPOCHS = 200
VALID_SIZE = 512

# Regularization parameters
DROPOUT = 0.5              # Feed forward dropout
RDROPOUT = 0.2             # Recurrent dropout
L2R = 1e-6                 # L2 regularization factor
CLIPNORM = 200             # Norm clipping for gradients
