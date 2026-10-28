"""Explicit image splits; no company data, automatic downloads, or label inference."""

from pathlib import Path

import tensorflow as tf

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if not files:
        raise ValueError(f"No BMP/PNG/JPEG images found in: {directory}")
    return files


def image_dataset(files, image_shape, batch_size=4, training=False, seed=42, targets=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    files = list(files)
    if not files:
        raise ValueError("An image dataset cannot be empty")
    dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in files])
    if training:
        dataset = dataset.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)

    def decode(path):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(tf.cast(image, tf.float32), image_shape[:2]) / 255.0
        return (image, image) if targets else image

    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.map(decode, num_parallel_calls=1).batch(batch_size).with_options(options).prefetch(1)
