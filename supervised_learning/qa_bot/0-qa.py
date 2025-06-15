#!/usr/bin/env python3
"""
0. Question Answering
"""

import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question

    Args:
        question (str): The question to answer
        reference (str): The reference document from which to find the answer

    Returns:
        str: The answer string, or None if no answer is found
    """
    try:
        # Load the pre-trained BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            'bert-large-uncased-whole-word-masking-finetuned-squad'
        )

        # Load the TensorFlow BERT model for question answering
        model = TFBertForQuestionAnswering.from_pretrained(
            'bert-large-uncased-whole-word-masking-finetuned-squad'
        )

        # Tokenize the question and reference text
        inputs = tokenizer.encode_plus(
            question,
            reference,
            add_special_tokens=True,
            return_tensors='tf',
            max_length=512,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True
        )

        # Extract input tensors
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # Run inference through the model
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get the most likely start and end positions
        start_index = tf.argmax(start_logits, axis=1).numpy()[0]
        end_index = tf.argmax(end_logits, axis=1).numpy()[0]

        # Check if we have a valid answer span
        if start_index >= end_index or start_index == 0:
            return None

        # Extract the answer tokens
        answer_tokens = input_ids[0][start_index:end_index + 1]

        # Decode the answer tokens back to text
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Return None if answer is empty or just whitespace
        if not answer or not answer.strip():
            return None

        return answer.strip()

    except Exception:
        return None
