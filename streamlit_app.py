import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2
import tempfile
import os
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from src.freshness_inference.model import FreshnessClassifier
from src.freshness_inference.constants import IMAGE_CONFIG, DETECTION_CONFIG
from src.freshness_inference.logging_config import setup_logging
from src.streamlit_ui.components import (
    render_image_upload,
    render_model_selection,
    render_confidence_slider,
    display_detection_results
)

# Setup logging
logger = setup_logging()

st.set_page_config(page_title="Freshness Detector", layout="wide")
st.title("Freshness Detector")
st.write("Take a picture or upload an image, then run YOLO detection.")

MODEL_GROUPS = {
    "Fruit & Veg": {
        "Light": {
            "path": "weights/best3.pt",
            "details": "Combined fruit-and-vegetable dataset — YOLO11m with 50 epochs, 768 image size, batch 16, and augmentation.",
        },
        "Heavy": {
            "path": "weights/best4-all-heavy.pt",
            "details": "Combined fruit-and-vegetable dataset — heavier combined-data model trained for 120 epochs at 896 resolution with AdamW, cosine LR, caching, and augmentation.",
        },
    },
    "Banana Only": {
        "Light": {
            "path": "weights/best.pt",
            "details": "Banana-only dataset — heavier YOLO11x banana-only run trained with 150 epochs, 896 image size, batch 16, and AdamW/cosine LR tuning.",
        },
        "Heavy": {
            "path": "weights/banana_yolo11n_best.pt",
            "details": "Banana-only dataset — lighter YOLO11n banana-only model trained with 100 epochs, 640 image size, and batch 32.",
        },
    },
}

@contextmanager
def temporary_image_file(pil_img: Image.Image):
    """
    Context manager for safe temporary file handling.
    
    Args:
        pil_img: PIL Image to save
        
    Yields:
        Path to temporary file
        
    Raises:
        IOError: If unable to create or save temporary file
    """
    tmp_file = None
    temp_path = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_path = tmp_file.name
        tmp_file.close()
        pil_img.save(temp_path, format="JPEG")
        yield temp_path
    except IOError as e:
        logger.error(f"Failed to create temporary image file: {e}")
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Load YOLO model with caching."""
    logger.info(f"Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model {model_path}: {e}")
        raise

@st.cache_resource
def load_freshness_classifier() -> FreshnessClassifier:
    """Load freshness classifier with caching."""
    logger.info("Loading freshness classifier")
    try:
        classifier = FreshnessClassifier()
        logger.info("Freshness classifier loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"Failed to load freshness classifier: {e}")
        raise

def extract_detections(yolo_results) -> List[Dict[str, Any]]:
    """
    Extract detection information from YOLO results.
    
    Args:
        yolo_results: YOLO prediction results
        
    Returns:
        List of Detection dictionaries
    """
    detections = []
    for box in yolo_results.boxes:
        cls_id = int(box.cls[0].item())
        conf_score = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        
        detections.append({
            "class": yolo_results.names[cls_id],
            "confidence": round(conf_score, 3),
            "x1": round(xyxy[0], 1),
            "y1": round(xyxy[1], 1),
            "x2": round(xyxy[2], 1),
            "y2": round(xyxy[3], 1)
        })
    return detections

def process_detections(
    detections: List[Dict[str, Any]],
    annotated_rgb: Image.Image,
    classifier: FreshnessClassifier
) -> tuple:
    """
    Process detections to extract resized images and predictions.
    
    Returns:
        Tuple of (resized_images, prediction_rows)
    """
    target_size = IMAGE_CONFIG.TARGET_SIZE
    resized_images = []
    prediction_rows = []
    
    for detection in detections:
        name = detection['class']
        xmin, ymin, xmax, ymax = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        
        # Convert float coordinates to integers for slicing
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        try:
            # Extract (Crop)
            crop = annotated_rgb[ymin:ymax, xmin:xmax]
            # Resize to target size
            resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            resized_images.append((name, resized))
            
            # Predict freshness
            prediction = classifier.predict_array(resized, image_path=f"detection_{len(resized_images)-1}")
            row = prediction.to_dict()
            row["detection_name"] = name
            row["fresh_probability"] = round(row["fresh_probability"], 2)
            row["rotten_probability"] = round(row["rotten_probability"], 2)
            prediction_rows.append(row)
            
        except Exception as e:
            logger.error(f"Failed to process detection {name}: {e}")
            # Continue with other detections
    
    return resized_images, prediction_rows

def determine_final_result(prediction_rows: List[Dict[str, Any]]) -> tuple:
    """
    Determine the final freshness result.
    
    Returns:
        Tuple of (final_label, final_prob)
    """
    if not prediction_rows:
        return "No predictions", 0.0
        
    rotten_preds = [p for p in prediction_rows if p["predicted_label"] == "Rotten"]
    if rotten_preds:
        final_pred = max(rotten_preds, key=lambda p: p["rotten_probability"])
        return "Rotten", final_pred["rotten_probability"]
    else:
        final_pred = max(prediction_rows, key=lambda p: p["fresh_probability"])
        return "Fresh", final_pred["fresh_probability"]

def draw_final_image(
    pil_img: Image.Image,
    detections: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]]
) -> Image.Image:
    """
    Draw bounding boxes and predictions on the final image.
    
    Returns:
        Image with annotations
    """
    final_image = pil_img.copy()
    draw = ImageDraw.Draw(final_image)
    font = ImageFont.load_default()
    
    for idx, detection in enumerate(detections):
        if idx >= len(prediction_rows):
            continue
            
        pred = prediction_rows[idx]
        label = pred["predicted_label"]
        prob = pred["fresh_probability"] if label == "Fresh" else pred["rotten_probability"]
        text = f"{label} {prob * 100:.2f}%"
        x1 = int(detection["x1"])
        y1 = int(detection["y1"])
        x2 = int(detection["x2"])
        y2 = int(detection["y2"])
        box_color = (0, 255, 0) if label == "Fresh" else (255, 0, 0)
        text_bg_color = (0, 0, 255)
        
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x1
        text_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            [
                (text_x, text_y),
                (text_x + text_width + 6, text_y + text_height + 4),
            ],
            fill=text_bg_color,
        )
        draw.text((text_x + 3, text_y + 2), text, fill=(255, 255, 255), font=font)
    
    return final_image

# Main UI
selected_model_conf = render_model_selection(MODEL_GROUPS)
model = load_model(selected_model_conf["path"])
classifier = load_freshness_classifier()

pil_img = render_image_upload()
conf = render_confidence_slider()

if pil_img is not None:
    # Process image
    try:
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        pil_img = pil_img.resize(IMAGE_CONFIG.PREVIEW_SIZE)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original image")
            st.image(pil_img, width="stretch")
        
        # Run detection
        with st.spinner("Running detection..."):
            try:
                with temporary_image_file(pil_img) as temp_path:
                    results = model.predict(source=temp_path, conf=conf, verbose=False)
                    r = results[0]
                    
                    annotated_bgr = r.plot()
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    
                    if r.boxes is None or len(r.boxes) == 0:
                        st.write("No detections found.")
                    else:
                        detections = extract_detections(r)
                        resized_images, prediction_rows = process_detections(detections, annotated_rgb, classifier)
                        
                        final_label, final_prob = determine_final_result(prediction_rows)
                        final_caption = f"Final result: {final_label} ({final_prob * 100:.2f}%)"
                        
                        # Display results
                        display_detection_results(detections, prediction_rows, annotated_rgb, resized_images)
                        
                        final_image = draw_final_image(pil_img, detections, prediction_rows)
                        
                        with col2:
                            st.subheader("Prediction")
                            st.image(final_image, width="stretch")
                            
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                st.error(f"Detection failed: {str(e)}")
                
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        st.error(f"Image processing failed: {str(e)}")