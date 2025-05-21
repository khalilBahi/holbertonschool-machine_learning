#!/usr/bin/env python3
"""3. Extract Word2Vec"""

import tf_utils  # This will suppress TensorFlow warnings
import tensorflow.keras as keras  # type: ignore


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer.

    Parameters:
    model: A trained gensim word2vec model

    Returns:
    Embedding: A trainable keras Embedding layer initialized with the
               word2vec model weights
    """

    # Get the vocabulary size and embedding dimension from the gensim model
    vocab_size = len(model.wv)
    embedding_dim = model.vector_size

    # Create a Keras Embedding layer
    # Set trainable=True so the weights can be further updated in Keras
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[model.wv.vectors],
        trainable=True
    )

    return embedding_layer
