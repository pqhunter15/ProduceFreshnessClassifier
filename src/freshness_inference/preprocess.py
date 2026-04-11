from pathlib import Path
from typing import Tuple, Optional
import logging
import os

import numpy as np
import tensorflow as tf

from .constants import IMAGE_CONFIG

logger = logging.getLogger(__name__)

def validate_image_path(image_path: str) -> Path:
    """
    Validate that image path exists and is readable.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file isn't readable
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")
    
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Image file is not readable: {image_path}")
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    
    logger.debug(f"Image path validated: {image_path}")
    return path

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = IMAGE_CONFIG.TARGET_SIZE) -> tf.Tensor:
    """
    Load and preprocess image for inference.
    
    Args:
        image_path: Path to image file
        target_size: Target dimensions (height, width)
        
    Returns:
        Preprocessed image tensor of shape (224, 224, 3)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image is corrupted or invalid format
    """
    path = validate_image_path(image_path)
    
    try:
        img_bytes = tf.io.read_file(str(path))
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image (H, W, 3), got shape {img.shape}"
            )
        
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape((target_size[0], target_size[1], 3))
        
        logger.debug(f"Successfully preprocessed image: {image_path}")
        return img
        
    except tf.errors.InvalidArgumentError as e:
        logger.error(f"Failed to decode image {image_path}: {e}")
        raise ValueError(f"Corrupted or invalid image file: {image_path}") from e

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
    if tuple(img.shape[:2]) != IMAGE_CONFIG.TARGET_SIZE:
        img = tf.image.resize(img, IMAGE_CONFIG.TARGET_SIZE)

    img.set_shape((IMAGE_CONFIG.TARGET_SIZE[0], IMAGE_CONFIG.TARGET_SIZE[1], 3))
    return tf.expand_dims(img, axis=0)
