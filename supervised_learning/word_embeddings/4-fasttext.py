#!/usr/bin/env python3
"""4. FastText"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim fastText model.

    Parameters:
    sentences (list): A list of sentences to be trained on
    vector_size (int): The dimensionality of the embedding layer
    min_count (int): The minimum number of
    occurrences of a word for use in training
    window (int): The maximum distance between the
    current and predicted word within a sentence
    negative (int): The size of negative sampling
    cbow (bool): A boolean to determine the training type;
                 True is for CBOW; False is for Skip-gram
    epochs (int): The number of iterations to train over
    seed (int): The seed for the random number generator
    workers (int): The number of worker threads to train the model

    Returns:
    FastText: The trained model
    """
    # Set the training algorithm based on cbow parameter
    # In gensim, sg=0 is CBOW and sg=1 is Skip-gram
    sg = 0 if cbow else 1

    # Create and train the FastText model
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
