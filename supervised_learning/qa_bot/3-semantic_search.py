#!/usr/bin/env python3
"""
3. Semantic Search
"""

import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents

    Args:
        corpus_path (str): Path to the corpus of reference documents
        sentence (str): The sentence from which to perform semantic search

    Returns:
        str: The reference text of the document most similar to sentence
    """
    # Read all documents from the corpus
    documents = []
    document_texts = []

    if not os.path.exists(corpus_path):
        return ""

    # Load all markdown files from the corpus directory
    for filename in sorted(os.listdir(corpus_path)):
        if filename.endswith('.md'):
            file_path = os.path.join(corpus_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty documents
                        documents.append(filename)
                        document_texts.append(content)
            except Exception:
                continue

    if not document_texts:
        return ""

    try:
        # Use a lightweight BERT model for embeddings
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModel.from_pretrained(model_name)

        def get_embeddings(texts):
            """Get embeddings for a list of texts"""
            embeddings = []
            for text in texts:
                # Tokenize and truncate to max length
                inputs = tokenizer(
                    text,
                    return_tensors='tf',
                    max_length=512,
                    truncation=True,
                    padding=True)

                # Get model outputs
                outputs = model(**inputs)

                # Use mean pooling of last hidden states
                embeddings.append(
                    tf.reduce_mean(
                        outputs.last_hidden_state,
                        axis=1))

            return tf.concat(embeddings, axis=0)

        # Get embeddings for query and documents
        query_embedding = get_embeddings([sentence])
        document_embeddings = get_embeddings(document_texts)

        # Calculate cosine similarities
        similarities = cosine_similarity(
            query_embedding.numpy(),
            document_embeddings.numpy()
        )[0]

        # Find the index of the most similar document
        most_similar_idx = np.argmax(similarities)

        # Return the text of the most similar document
        return document_texts[most_similar_idx]

    except Exception:
        # Fallback to TF-IDF if transformer models fail
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000
        )

        # Combine query and documents for vectorization
        all_texts = [sentence] + document_texts

        # Fit and transform all texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Get the query vector (first row)
        query_vector = tfidf_matrix[0:1]

        # Get document vectors (remaining rows)
        document_vectors = tfidf_matrix[1:]

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, document_vectors)[0]

        # Find the index of the most similar document
        most_similar_idx = np.argmax(similarities)

        # Return the text of the most similar document
        return document_texts[most_similar_idx]
