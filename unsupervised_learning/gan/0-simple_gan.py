#!/usr/bin/env python3
"""Simple Wasserstein GAN (WGAN) with weight clipping"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """Wasserstein GAN training loop with weight clipping on the critic.

    Parameters
    - generator: tf.keras.Model
        The generator G mapping latent vectors to samples in data space.
    - discriminator: tf.keras.Model
        The critic/discriminator D mapping samples to a scalar score.
    - latent_generator: Callable[[int], tf.Tensor]
        Function that returns a batch of latent
        vectors when called with a size.
    - real_examples: tf.Tensor
        Tensor containing real training examples to sample from.
    - batch_size: int, default 200
        Number of samples per batch for both real and fake.
    - disc_iter: int, default 1
        Number of critic updates per generator update.
    - learning_rate: float, default 0.005
        Learning rate for both generator and discriminator Adam optimizers.

    Notes
    - Generator loss: L_G = -E[D(G(z))]
    - Critic loss:    L_D = E[D(fake)] - E[D(real)] (minimize w.r.t. D)
    - Weights of the critic are clipped to [-1, 1] after each update.
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=1,
        learning_rate=0.005,
    ):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            clipnorm=1.0,
        )
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: tf.math.reduce_mean(
            y
        ) - tf.math.reduce_mean(x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            clipnorm=1.0,
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """Return a batch of generated (fake) samples.

        Parameters
        - size: Optional[int]
            Batch size to generate. Defaults to `self.batch_size`.
        - training: bool
            Forward pass flag for layers like BatchNorm/Dropout.

        Returns
        - tf.Tensor: Generated samples of shape compatible with real examples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """Return a random batch of real samples drawn without replacement.

        Parameters
        - size: Optional[int]
            Batch size to draw. Defaults to `self.batch_size`.

        Returns
        - tf.Tensor: A batch of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        """Run one training step of WGAN with weight clipping.

        This method performs `disc_iter` critic updates followed by one
        generator update. It returns the latest discriminator and generator
        losses for logging.

        Parameters
        - useless_argument: Any
            Placeholder to satisfy Keras `train_step` signature. Ignored.

        Returns
        - dict: {"discr_loss": tf.Tensor, "gen_loss": tf.Tensor}
        """
        # Discriminator training
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                # Get real and fake samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                # Compute discriminator outputs
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)

                # Compute discriminator loss
                discr_loss = self.discriminator.loss(fake_output, real_output)

            # Compute gradients and apply to discriminator
            disc_gradients = disc_tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(disc_gradients, self.discriminator.trainable_variables)
            )

            # Clip the weights of the discriminator between -1 and 1
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # Generator training
        with tf.GradientTape() as gen_tape:
            # Get fake samples
            fake_samples = self.get_fake_sample(training=True)

            # Compute discriminator output on fake samples
            fake_output = self.discriminator(fake_samples, training=True)

            # Compute generator loss
            gen_loss = self.generator.loss(fake_output)

        # Compute gradients and apply to generator
        gen_gradients = gen_tape.gradient(gen_loss,
                                          self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
