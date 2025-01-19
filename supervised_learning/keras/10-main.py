#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 8

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# Imports
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':
    # Load the first model
    network = model.load_model('network2.keras')
    
    # Save the weights of the first model with the correct file extension
    weights.save_weights(network, 'weights2.weights.h5')  # Use `.weights.h5` extension
    del network  # Delete the model to free up memory

    # Load the second model
    network2 = model.load_model('network1.keras')
    
    # Print the initial weights of the second model
    print(network2.get_weights())
    
    # Load the saved weights into the second model with the correct file extension
    weights.load_weights(network2, 'weights2.weights.h5')  # Use `.weights.h5` extension
    
    # Print the weights of the second model after loading
    print(network2.get_weights())