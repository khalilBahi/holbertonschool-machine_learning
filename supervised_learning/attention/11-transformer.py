#!/usr/bin/env python3
"""10. Transformer Network"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformer class that inherits from tensorflow.keras.Model"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initialize the Transformer

        Args:
            N (int): Number of blocks in the encoder and decoder
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in the fully connected layers
            input_vocab (int): Size of the input vocabulary
            target_vocab (int): Size of the target vocabulary
            max_seq_input (int): Maximum sequence length possible for the input
            max_seq_target (int): Maximum sequence length possible
                for the target
            drop_rate (float): Dropout rate
        """
        super(Transformer, self).__init__()

        # Create encoder
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate)

        # Create decoder
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate)

        # Final linear layer to project to target vocabulary
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
            self,
            inputs,
            target,
            training=None,
            encoder_mask=None,
            look_ahead_mask=None,
            decoder_mask=None):
        """
        Forward pass through the transformer

        Args:
            inputs: tensor of shape (batch, input_seq_len)
                containing the inputs
            target: tensor of shape (batch, target_seq_len)
                containing the target
            training: boolean to determine if the model is training
            encoder_mask: padding mask to be applied to the encoder
            look_ahead_mask: look ahead mask to be applied to the decoder
            decoder_mask: padding mask to be applied to the decoder

        Returns:
            tensor of shape (batch, target_seq_len, target_vocab)
                containing transformer output
        """
        # Pass inputs through encoder
        encoder_output = self.encoder(
            inputs, training=training, mask=encoder_mask)

        # Pass target and encoder output through decoder
        decoder_output = self.decoder(
            target,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask)

        # Apply final linear layer to get output logits
        final_output = self.linear(decoder_output)

        return final_output
