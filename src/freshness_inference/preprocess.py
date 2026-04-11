from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import IMG_SIZE


def load_and_preprocess_image(image_path: str) -> tf.Tensor:
    """
    Load one image from disk and preprocess it for inference.

    Returns:
        tf.Tensor of shape (224, 224, 3), dtype float32, values in [0, 1].
    """
    image_path = str(Path(image_path))

    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


def make_batch_from_path(image_path: str) -> tf.Tensor:
    """
    Convert a single image path into a batch of size 1.
    """
    img = load_and_preprocess_image(image_path)
    return tf.expand_dims(img, axis=0)


def make_batch_from_array(image_array: np.ndarray) -> tf.Tensor:
    """
    Convert a single RGB numpy image into a batch of size 1.
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must be an RGB image with shape (H, W, 3)")

    img = image_array.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    if tuple(img.shape[:2]) != IMG_SIZE:
        img = tf.image.resize(img, IMG_SIZE)

    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return tf.expand_dims(img, axis=0)
