"""Portable TensorFlow builders adapted from the thesis AE notebooks.

Image loading and RGB scaling to [0, 1] belong to the caller. No application
preprocessing is inserted, and ImageNet weights are downloaded only when the
caller explicitly passes ``weights="imagenet"``.
"""

import math
from numbers import Integral

import tensorflow as tf


_BACKBONES = {
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
    "resnet50": tf.keras.applications.ResNet50,
}


def _validate_image_shape(image_shape):
    """Require fixed RGB dimensions that support all six downsampling stages."""
    try:
        shape = tuple(image_shape)
    except TypeError as exc:
        raise ValueError("image_shape must contain height, width, and 3 channels") from exc
    if len(shape) != 3 or any(
        isinstance(dimension, bool) or not isinstance(dimension, Integral)
        for dimension in shape
    ):
        raise ValueError("image_shape must contain three fixed integer dimensions")
    shape = tuple(int(dimension) for dimension in shape)
    if shape[0] < 64 or shape[1] < 64 or shape[2] != 3:
        raise ValueError("image_shape requires height and width >= 64 and 3 RGB channels")
    return shape


def _validate_latent_dim(latent_dim):
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, Integral) or latent_dim < 1:
        raise ValueError("latent_dim must be a positive integer")
    return int(latent_dim)


def build_feature_extractor(
    image_shape=(400, 320, 3), backbone="cnn", weights=None, pooling="flatten"
):
    """Return CNN features or features from a frozen application backbone.

    The CNN preserves the original six same-padded convolution/pooling stages
    and always flattens their output. Application backbones optionally use
    global average pooling for the VAE adapter; AE defaults retain flattening.
    ``weights=None`` creates an offline, randomly initialized backbone, still
    frozen; it does not imply ImageNet pretraining.
    """
    image_shape = _validate_image_shape(image_shape)
    if not isinstance(backbone, str) or backbone.lower() not in ("cnn", *_BACKBONES):
        raise ValueError("backbone must be cnn, mobilenetv2, inceptionresnetv2, or resnet50")
    backbone = backbone.lower()
    if pooling not in ("flatten", "avg"):
        raise ValueError("pooling must be 'flatten' or 'avg'")
    if backbone == "cnn" and weights is not None:
        raise ValueError("the thesis CNN has no pretrained weights; use weights=None")
    if backbone == "inceptionresnetv2" and min(image_shape[:2]) < 75:
        raise ValueError("InceptionResNetV2 requires height and width >= 75")

    inputs = tf.keras.Input(shape=image_shape, name="image")
    if backbone == "cnn":
        features = inputs
        for index, filters in enumerate((128, 64, 64, 64, 32, 32), start=1):
            features = tf.keras.layers.Conv2D(
                filters, 3, activation="relu", padding="same", name=f"encoder_conv_{index}"
            )(features)
            features = tf.keras.layers.MaxPooling2D(
                2, padding="same", name=f"encoder_pool_{index}"
            )(features)
        features = tf.keras.layers.Flatten(name="flatten_features")(features)
    else:
        application = _BACKBONES[backbone](
            include_top=False, weights=weights, input_shape=image_shape
        )
        application.trainable = False
        features = application(inputs, training=False)
        if pooling == "avg":
            features = tf.keras.layers.GlobalAveragePooling2D(name="average_features")(features)
        else:
            features = tf.keras.layers.Flatten(name="flatten_features")(features)
    return tf.keras.Model(inputs, features, name=f"{backbone}_feature_extractor")


def build_encoder(image_shape=(400, 320, 3), latent_dim=512, backbone="cnn", weights=None):
    """Build the AE encoder with the original ReLU dense bottleneck."""
    latent_dim = _validate_latent_dim(latent_dim)
    feature_extractor = build_feature_extractor(image_shape, backbone, weights)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent_vector")(
        feature_extractor.output
    )
    return tf.keras.Model(feature_extractor.input, latent, name="encoder")


def build_decoder(image_shape=(400, 320, 3), latent_dim=512):
    """Build the six-stage decoder, preserving the original default geometry.

    At 400 x 320 the seed is 8 x 10 x 64 and cropping is 56 pixels per vertical
    side and 160 per horizontal side. Other dimensions use the smallest seed
    covering the target after six doublings, with near-symmetric cropping.
    """
    image_shape = _validate_image_shape(image_shape)
    latent_dim = _validate_latent_dim(latent_dim)
    height, width, channels = image_shape
    if image_shape == (400, 320, 3):
        seed_height, seed_width = 8, 10
    else:
        seed_height, seed_width = math.ceil(height / 64), math.ceil(width / 64)

    inputs = tf.keras.Input(shape=(latent_dim,), name="latent_input")
    reconstruction = tf.keras.layers.Dense(
        seed_height * seed_width * 64, activation="relu", name="decoder_dense"
    )(inputs)
    reconstruction = tf.keras.layers.Reshape(
        (seed_height, seed_width, 64), name="decoder_seed"
    )(reconstruction)
    for index, filters in enumerate((256, 128, 64, 64, 32, 16), start=1):
        reconstruction = tf.keras.layers.Conv2DTranspose(
            filters,
            3,
            strides=2,
            activation="relu",
            padding="same",
            name=f"decoder_upsample_{index}",
        )(reconstruction)
    reconstruction = tf.keras.layers.Conv2D(
        channels, 3, activation="sigmoid", padding="same", name="rgb_reconstruction"
    )(reconstruction)

    excess_height = seed_height * 64 - height
    excess_width = seed_width * 64 - width
    if excess_height or excess_width:
        reconstruction = tf.keras.layers.Cropping2D(
            (
                (excess_height // 2, excess_height - excess_height // 2),
                (excess_width // 2, excess_width - excess_width // 2),
            ),
            name="crop_to_input_shape",
        )(reconstruction)
    return tf.keras.Model(inputs, reconstruction, name="decoder")


def build_autoencoder(image_shape=(400, 320, 3), latent_dim=512, backbone="cnn", weights=None):
    """Return ``(encoder, decoder, autoencoder)`` without compiling or training."""
    encoder = build_encoder(image_shape, latent_dim, backbone, weights)
    decoder = build_decoder(image_shape, latent_dim)
    autoencoder = tf.keras.Model(
        encoder.input, decoder(encoder.output), name="autoencoder"
    )
    return encoder, decoder, autoencoder
