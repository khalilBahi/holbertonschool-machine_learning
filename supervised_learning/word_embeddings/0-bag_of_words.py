#!/usr/bin/env python3
"""0. Bag Of Words"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    sentences (list): A list of sentences to analyze
    vocab (list, optional): A list of vocabulary words to use for the analysis.
                           If None, all words within sentences should be used.

    Returns:
    tuple: (embeddings, features)
        - embeddings: numpy.ndarray of shape (s, f) containing the embeddings
          where s is the number of sentences and f is the number of features
        - features: list of the features used for embeddings
    """
    # Process sentences to extract words
    processed_sentences = []
    all_words = set()

    for sentence in sentences:
        # Convert to lowercase and remove punctuation
        # Handle apostrophes specially to remove possessives
        clean_sentence = sentence.lower()
        clean_sentence = re.sub(
            r"'s\b", "", clean_sentence)  # Remove possessive 's
        # Remove other punctuation
        clean_sentence = re.sub(r'[^\w\s]', '', clean_sentence)
        words = clean_sentence.split()
        processed_sentences.append(words)

        # Add words to the set of all words if vocab is None
        if vocab is None:
            all_words.update(words)

    # Create features list (vocabulary)
    if vocab is None:
        features = np.array(sorted(list(all_words)))
    else:
        features = np.array(sorted(vocab))

    # Create embedding matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Create word-to-index mapping for faster lookup
    word_to_idx = {word: idx for idx, word in enumerate(features.tolist())}

    # Fill the embedding matrix
    for i, words in enumerate(processed_sentences):
        for word in words:
            if word in word_to_idx:  # Only count words in our vocabulary
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, features
