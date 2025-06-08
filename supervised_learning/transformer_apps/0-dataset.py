#!/usr/bin/env python3
""""0. Dataset"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from transformers import BertTokenizerFast


class Dataset:
    """Dataset class that loads and preps a dataset for machine translation"""

    def __init__(self):
        """
        Class constructor that creates instance attributes:
        - data_train: ted_hrlr_translate/pt_to_en tf.data.Dataset train split
        - data_valid: ted_hrlr_translate/pt_to_en tf.data.Dataset valid split
        - tokenizer_pt: Portuguese tokenizer created from training set
        - tokenizer_en: English tokenizer created from training set
        """
        # Load the dataset splits
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        # Create tokenizers from the training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)
                pt: tf.Tensor containing Portuguese sentence
                en: tf.Tensor containing corresponding English sentence

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Load base pre-trained tokenizers
        base_tokenizer_pt = BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        base_tokenizer_en = BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        # Collect text data for training tokenizers
        pt_texts = []
        en_texts = []

        # Extract text from the dataset
        for pt, en in data:
            pt_texts.append(pt.numpy().decode('utf-8'))
            en_texts.append(en.numpy().decode('utf-8'))

        # Train tokenizers with maximum vocabulary size of 2**13 (8192)
        vocab_size = 2**13

        # Train Portuguese tokenizer
        tokenizer_pt = base_tokenizer_pt.train_new_from_iterator(
            pt_texts,
            vocab_size=vocab_size
        )

        # Train English tokenizer
        tokenizer_en = base_tokenizer_en.train_new_from_iterator(
            en_texts,
            vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens: np.ndarray containing the Portuguese tokens
            en_tokens: np.ndarray containing the English tokens
        """
        # Decode tensors to strings
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Tokenize the sentences
        pt_tokens = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_text, add_special_tokens=False)

        # Get vocabulary size (2**13 = 8192)
        vocab_size = 2**13

        # Add start and end tokens
        # Start token: vocab_size (8192)
        # End token: vocab_size + 1 (8193)
        pt_tokens = [vocab_size] + pt_tokens + [vocab_size + 1]
        en_tokens = [vocab_size] + en_tokens + [vocab_size + 1]

        # Convert to numpy arrays
        pt_tokens = np.array(pt_tokens)
        en_tokens = np.array(en_tokens)

        return pt_tokens, en_tokens
