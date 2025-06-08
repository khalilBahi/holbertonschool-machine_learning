#!/usr/bin/env python3
"""3. Pipeline"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from transformers import BertTokenizerFast


class Dataset:
    """Dataset class that loads and preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """
        Class constructor that creates instance attributes:
        - data_train: ted_hrlr_translate/pt_to_en tf.data.Dataset train split
        - data_valid: ted_hrlr_translate/pt_to_en tf.data.Dataset valid split
        - tokenizer_pt: Portuguese tokenizer created from training set
        - tokenizer_en: English tokenizer created from training set

        Args:
            batch_size: batch size for training/validation
            max_len: maximum number of tokens allowed per example sentence
        """
        self.batch_size = batch_size
        self.max_len = max_len
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

        # Tokenize the datasets using tf_encode
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Set up training data pipeline
        self.data_train = self.data_train.filter(self._filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            self.batch_size, padded_shapes=(
                [None], [None]), padding_values=(
                tf.constant(
                    0, dtype=tf.int64), tf.constant(
                    0, dtype=tf.int64)))
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        # Set up validation data pipeline
        self.data_valid = self.data_valid.filter(self._filter_max_len)
        self.data_valid = self.data_valid.padded_batch(
            self.batch_size, padded_shapes=(
                [None], [None]), padding_values=(
                tf.constant(
                    0, dtype=tf.int64), tf.constant(
                    0, dtype=tf.int64)))

    def _filter_max_len(self, pt, en):
        """
        Filter function to remove examples with sentences longer than max_len

        Args:
            pt: tf.Tensor containing Portuguese tokens
            en: tf.Tensor containing English tokens

        Returns:
            bool: True if both sentences are <= max_len, False otherwise
        """
        return tf.logical_and(
            tf.size(pt) <= self.max_len,
            tf.size(en) <= self.max_len
        )

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

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens: tf.Tensor containing Portuguese tokens with proper shape
            en_tokens: tf.Tensor containing English tokens with proper shape
        """
        # Use tf.py_function to wrap the encode method
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set the shape of the tensors (they will be 1D with unknown length)
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
