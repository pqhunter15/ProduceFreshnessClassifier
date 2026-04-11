"""Application configuration constants."""

from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ImageConfig:
    """Image processing configuration."""
    TARGET_HEIGHT: int = 224
    TARGET_WIDTH: int = 224
    TARGET_SIZE: Tuple[int, int] = (224, 224)
    
    # For display/preview
    PREVIEW_HEIGHT: int = 720
    PREVIEW_WIDTH: int = 1280
    PREVIEW_SIZE: Tuple[int, int] = (1280, 720)
    
@dataclass(frozen=True)
class DetectionConfig:
    """YOLO detection configuration."""
    DEFAULT_CONFIDENCE: float = 0.60
    MIN_CONFIDENCE: float = 0.0
    MAX_CONFIDENCE: float = 1.0
    CONFIDENCE_STEP: float = 0.05
    
@dataclass(frozen=True)
class ClassificationConfig:
    """Freshness classification configuration."""
    DEFAULT_THRESHOLD: float = 0.5
    FRESH_CLASS_ID: int = 0
    ROTTEN_CLASS_ID: int = 1

# Global instances
IMAGE_CONFIG = ImageConfig()
DETECTION_CONFIG = DetectionConfig()
CLASSIFICATION_CONFIG = ClassificationConfig()