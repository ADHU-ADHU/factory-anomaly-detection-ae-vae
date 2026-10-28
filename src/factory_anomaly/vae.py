"""Modern TensorFlow reference adapter for the thesis VAE experiments.

This is a maintained implementation of the experimental design, not a verbatim
replication of any notebook or a reproduction of its reported results. It uses
one consistent objective: ``reconstruction_weight * mean_pixel_MSE + KL``.
Inference reconstructs from the posterior mean; training samples the posterior.
Input images are RGB floating point values in [0, 1], as in the source loaders.
"""

import math
from numbers import Real

import tensorflow as tf

from .models import (
    _validate_image_shape,
    _validate_latent_dim,
    build_decoder,
    build_feature_extractor,
)


@tf.keras.utils.register_keras_serializable(package="factory_anomaly")
class SamplingLayer(tf.keras.layers.Layer):
    """Reparameterize a diagonal Gaussian using its mean and log variance."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(tf.shape(z_mean), dtype=z_mean.dtype)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@tf.keras.utils.register_keras_serializable(package="factory_anomaly")
class VAE(tf.keras.Model):
    """VAE with serialized component models and an explicit custom objective.

    ``encoder`` must return ``[z_mean, z_log_var]``. The decoder must reconstruct
    the encoder's image input shape. Both ``fit(x)`` and ``fit(x, x)`` are
    supported: targets are ignored because the reconstruction target is the
    input itself. Sample weighting is deliberately unsupported.
    """

    def __init__(self, encoder, decoder, reconstruction_weight=500.0, **kwargs):
        super().__init__(**kwargs)
        if (
            isinstance(reconstruction_weight, bool)
            or not isinstance(reconstruction_weight, Real)
            or not math.isfinite(reconstruction_weight)
            or reconstruction_weight <= 0
        ):
            raise ValueError("reconstruction_weight must be a finite positive number")
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_weight = float(reconstruction_weight)
        self.sampling = SamplingLayer(name="posterior_sampling")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """Expose resettable epoch metrics to Keras ``fit`` and ``evaluate``."""
        return [
            self.loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def build(self, input_shape):
        if not self.encoder.built:
            self.encoder.build(input_shape)
        if not self.decoder.built:
            self.decoder.build(self.encoder.output_shape[0])
        super().build(input_shape)

    def call(self, inputs, training=False, sample=None):
        """Reconstruct images; sampling is enabled for training only by default.

        Pass ``sample=True`` for an explicitly stochastic reconstruction outside
        training. Normal inference, including ``predict``, uses ``z_mean``.
        """
        z_mean, z_log_var = self.encoder(inputs, training=training)
        if sample is None:
            sample = False if training is None else training
        if isinstance(sample, bool):
            latent = self.sampling([z_mean, z_log_var]) if sample else z_mean
        else:
            latent = tf.cond(
                tf.cast(sample, tf.bool),
                lambda: self.sampling([z_mean, z_log_var]),
                lambda: z_mean,
            )
        return self.decoder(latent, training=training)

    @staticmethod
    def _images_from_data(data):
        images, _targets, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is not None:
            raise ValueError("VAE does not support sample weights")
        return images

    def _objective(self, images, training):
        z_mean, z_log_var = self.encoder(images, training=training)
        latent = self.sampling([z_mean, z_log_var]) if training else z_mean
        reconstruction = self.decoder(latent, training=training)
        # Accumulate the objective in float32, including when model layers use
        # a lower precision policy. MSE averages all pixels and RGB channels.
        images = tf.cast(images, tf.float32)
        reconstruction = tf.cast(reconstruction, tf.float32)
        z_mean = tf.cast(z_mean, tf.float32)
        z_log_var = tf.cast(z_log_var, tf.float32)
        reconstruction_loss = self.reconstruction_weight * tf.reduce_mean(
            tf.square(images - reconstruction)
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                axis=-1,
            )
        )
        return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss

    def _update_metrics(self, values, images):
        # Weight scalar batch losses by batch size so a short final batch does
        # not contribute as much to the epoch mean as a full batch.
        batch_size = tf.cast(tf.shape(images)[0], tf.float32)
        for tracker, value in zip(self.metrics, values):
            tracker.update_state(value, sample_weight=batch_size)
        return {tracker.name: tracker.result() for tracker in self.metrics}

    def train_step(self, data):
        images = self._images_from_data(data)
        with tf.GradientTape() as tape:
            values = self._objective(images, training=True)
        gradients = tape.gradient(values[0], self.trainable_variables)
        self.optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables)
            if gradient is not None
        )
        return self._update_metrics(values, images)

    def test_step(self, data):
        images = self._images_from_data(data)
        values = self._objective(images, training=False)
        return self._update_metrics(values, images)

    def get_config(self):
        config = super().get_config()
        config.update(
            encoder=tf.keras.utils.serialize_keras_object(self.encoder),
            decoder=tf.keras.utils.serialize_keras_object(self.decoder),
            reconstruction_weight=self.reconstruction_weight,
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        encoder = tf.keras.utils.deserialize_keras_object(config.pop("encoder"))
        decoder = tf.keras.utils.deserialize_keras_object(config.pop("decoder"))
        return cls(encoder=encoder, decoder=decoder, **config)


def build_vae(
    image_shape=(400, 320, 3),
    latent_dim=512,
    backbone="cnn",
    weights=None,
    reconstruction_weight=500.0,
):
    """Return ``(encoder, decoder, vae)`` without compiling or training.

    CNN features are flattened; frozen application features are globally
    averaged, matching the representative VAE notebook families. Application
    preprocessing is not inserted by this adapter. ``weights=None`` performs
    no download and does not represent pretrained features.
    """
    image_shape = _validate_image_shape(image_shape)
    latent_dim = _validate_latent_dim(latent_dim)
    pooling = "flatten" if isinstance(backbone, str) and backbone.lower() == "cnn" else "avg"
    feature_extractor = build_feature_extractor(
        image_shape, backbone=backbone, weights=weights, pooling=pooling
    )
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(feature_extractor.output)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(
        feature_extractor.output
    )
    encoder = tf.keras.Model(
        feature_extractor.input, [z_mean, z_log_var], name="variational_encoder"
    )
    decoder = build_decoder(image_shape=image_shape, latent_dim=latent_dim)
    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        reconstruction_weight=reconstruction_weight,
        name="variational_autoencoder",
    )
    # All weights live in the already-built functional component models. Mark
    # the wrapper built without allocating a full-resolution synthetic batch.
    vae.build((None, *image_shape))
    return encoder, decoder, vae
