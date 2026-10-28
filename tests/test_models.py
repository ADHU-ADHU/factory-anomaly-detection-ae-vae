"""Offline checks for the portable model adapters, using synthetic images only."""

from pathlib import Path
import tempfile
import unittest

import tensorflow as tf

from factory_anomaly.models import (
    build_autoencoder,
    build_decoder,
    build_feature_extractor,
)
from factory_anomaly.vae import build_vae


class ModelTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17)

    def tearDown(self):
        tf.keras.backend.clear_session()

    @staticmethod
    def _images():
        return tf.reshape(tf.linspace(0.0, 1.0, 64 * 64 * 3), (1, 64, 64, 3))

    def test_default_autoencoder_preserves_source_geometry_and_parameter_count(self):
        encoder, decoder, autoencoder = build_autoencoder()
        self.assertEqual(encoder.output_shape, (None, 512))
        self.assertEqual(decoder.output_shape, (None, 400, 320, 3))
        self.assertEqual(autoencoder.output_shape, (None, 400, 320, 3))
        self.assertEqual(autoencoder.count_params(), 3_956_451)
        self.assertEqual(decoder.get_layer("decoder_seed").output.shape[1:], (8, 10, 64))
        self.assertEqual(
            decoder.get_layer("crop_to_input_shape").cropping,
            ((56, 56), (160, 160)),
        )

    def test_small_autoencoder_gradient_and_serialization(self):
        _encoder, _decoder, autoencoder = build_autoencoder((64, 64, 3), latent_dim=4)
        images = self._images()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        with tf.GradientTape() as tape:
            reconstruction = autoencoder(images, training=True)
            loss = tf.reduce_mean(tf.square(images - reconstruction))
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        self.assertEqual(tuple(reconstruction.shape), tuple(images.shape))
        self.assertTrue(bool(tf.math.is_finite(loss)))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients))
        before_update = autoencoder.trainable_variables[-1].numpy().copy()
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
        self.assertTrue(
            bool(tf.reduce_any(autoencoder.trainable_variables[-1] != before_update))
        )

        expected = autoencoder(images, training=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autoencoder.keras"
            autoencoder.save(path)
            restored = tf.keras.models.load_model(path)
            tf.debugging.assert_near(restored(images, training=False), expected, atol=1e-6)

    def test_decoder_crops_nondefault_odd_geometry(self):
        decoder = build_decoder((65, 97, 3), latent_dim=4)
        self.assertEqual(decoder.output_shape, (None, 65, 97, 3))
        self.assertEqual(
            decoder.get_layer("crop_to_input_shape").cropping, ((31, 32), (15, 16))
        )

    def test_vae_gradient_mean_inference_and_serialization(self):
        encoder, decoder, vae = build_vae((64, 64, 3), latent_dim=4)
        images = self._images()
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        before_update = decoder.trainable_variables[-1].numpy().copy()
        metrics = vae.train_on_batch(images, images, return_dict=True)
        self.assertEqual(set(metrics), {"loss", "reconstruction_loss", "kl_loss"})
        for name, value in metrics.items():
            self.assertTrue(bool(tf.math.is_finite(value)), name)
        self.assertGreaterEqual(float(metrics["kl_loss"]), 0.0)
        self.assertTrue(bool(tf.reduce_any(decoder.trainable_variables[-1] != before_update)))

        mean, _log_variance = encoder(images, training=False)
        expected = decoder(mean, training=False)
        first = vae(images, training=False)
        tf.debugging.assert_near(first, expected, atol=1e-6)
        tf.debugging.assert_equal(first, vae(images, training=False))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vae.keras"
            vae.save(path)
            restored = tf.keras.models.load_model(path)
            tf.debugging.assert_near(restored(images, training=False), first, atol=1e-6)

    def test_mobilenet_features_are_frozen_and_offline(self):
        features = build_feature_extractor(
            (64, 64, 3), backbone="mobilenetv2", weights=None, pooling="avg"
        )
        self.assertEqual(features.output_shape, (None, 1280))
        self.assertEqual(len(features.trainable_variables), 0)
        result = features(self._images(), training=False)
        self.assertEqual(tuple(result.shape), (1, 1280))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(result))))

    def test_invalid_configuration_has_clear_errors(self):
        for shape in ((32, 64, 3), (64, 64, 1), (64, 64), (None, 64, 3)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "image_shape"):
                    build_feature_extractor(shape)
        with self.assertRaisesRegex(ValueError, "backbone"):
            build_feature_extractor(backbone="unknown")
        with self.assertRaisesRegex(ValueError, "InceptionResNetV2"):
            build_feature_extractor((64, 64, 3), backbone="inceptionresnetv2")
        with self.assertRaisesRegex(ValueError, "latent_dim"):
            build_decoder(latent_dim=0)
        with self.assertRaisesRegex(ValueError, "pooling"):
            build_feature_extractor(pooling="unknown")


if __name__ == "__main__":
    unittest.main()
