from dataclasses import dataclass
from typing import Dict, Any
import logging

import numpy as np
import tensorflow as tf

from .config import (
    MODEL_REPO_ID,
    MODEL_FILENAME,
    CLASS_NAMES,
    DEFAULT_THRESHOLD,
)
from .preprocess import make_batch_from_path, make_batch_from_array
from .logging_config import setup_logging

logger = setup_logging()


@dataclass
class PredictionResult:
    image_path: str
    predicted_class: int
    predicted_label: str
    fresh_probability: float
    rotten_probability: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "fresh_probability": self.fresh_probability,
            "rotten_probability": self.rotten_probability,
            "threshold": self.threshold,
        }


class FreshnessClassifier:
    def __init__(
        self,
        repo_id: str = MODEL_REPO_ID,
        filename: str = MODEL_FILENAME,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.threshold = threshold
        logger.info(f"Loading model from {repo_id}/{filename}")
        self.model = self._load_model()
        logger.info("Model loaded successfully")

    def _load_model(self) -> tf.keras.Model:
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        
        model_dir = Path(__file__).parent
        model_path = model_dir / self.filename
        
        if not model_path.exists():
            logger.info(f"Downloading model from {self.repo_id}/{self.filename}")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                model_path = Path(downloaded_path)
                logger.info(f"Model downloaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise
        
        logger.debug(f"Loading model from {model_path}")
        return tf.keras.models.load_model(str(model_path))

    def _predict_batch(self, batch: tf.Tensor, image_path: str) -> PredictionResult:
        # Model outputs sigmoid probability for class 1 = Rotten
        rotten_prob = float(self.model.predict(batch, verbose=0)[0][0])
        fresh_prob = 1.0 - rotten_prob

        pred_class = 1 if rotten_prob >= self.threshold else 0
        pred_label = CLASS_NAMES[pred_class]

        logger.debug(f"Prediction for {image_path}: {pred_label} (fresh: {fresh_prob:.3f}, rotten: {rotten_prob:.3f})")

        return PredictionResult(
            image_path=image_path,
            predicted_class=pred_class,
            predicted_label=pred_label,
            fresh_probability=round(fresh_prob, 6),
            rotten_probability=round(rotten_prob, 6),
            threshold=self.threshold,
        )

    def predict_image(self, image_path: str) -> PredictionResult:
        logger.debug(f"Predicting freshness for image: {image_path}")
        batch = make_batch_from_path(image_path)
        result = self._predict_batch(batch, image_path)
        logger.info(f"Prediction: {result.predicted_label} (confidence: {result.fresh_probability if result.predicted_label == 'Fresh' else result.rotten_probability})")
        return result

    def predict_array(self, image_array: np.ndarray, image_path: str = "<array>") -> PredictionResult:
        logger.debug(f"Predicting freshness for array input: {image_path}")
        batch = make_batch_from_array(image_array)
        result = self._predict_batch(batch, image_path)
        logger.info(f"Array prediction: {result.predicted_label} (confidence: {result.fresh_probability if result.predicted_label == 'Fresh' else result.rotten_probability})")
        return result
