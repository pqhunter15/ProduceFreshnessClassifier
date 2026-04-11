"""Reusable Streamlit UI components."""
import streamlit as st
from typing import Optional, List, Dict, Any
from PIL import Image
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def render_image_upload() -> Optional[Image.Image]:
    """
    Render image upload/camera widget.
    
    Returns:
        PIL Image or None if no image selected
    """
    source = st.radio(
        "Choose image source",
        ["Upload an image", "Take a picture"],
        horizontal=True
    )
    
    image_file = None
    if source == "Take a picture":
        image_file = st.camera_input("Take a picture")
    else:
        image_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"]
        )
    
    if image_file is None:
        return None
    
    try:
        image = Image.open(image_file).convert("RGB")
        logger.debug(f"Image loaded: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        st.error(f"Failed to load image: {str(e)}")
        return None

def render_model_selection(model_groups: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Render model selection UI.
    
    Args:
        model_groups: Dictionary of model groups and versions
        
    Returns:
        Selected model configuration
    """
    model_group = st.radio(
        "Select model group",
        list(model_groups.keys()),
        index=0,
        horizontal=True,
    )
    model_version = st.radio(
        "Select version",
        ["Light", "Heavy"],
        index=0,
        horizontal=True,
    )
    selected_model_conf = model_groups[model_group][model_version]
    st.caption(selected_model_conf["details"])
    return selected_model_conf

def render_confidence_slider() -> float:
    """
    Render confidence threshold slider.
    
    Returns:
        Selected confidence value
    """
    return st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05
    )

def display_detection_results(
    detections: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]],
    annotated_rgb: Image.Image,
    resized_images: List[tuple]
) -> None:
    """
    Display detection and prediction results in expandable section.
    
    Args:
        detections: List of detection dictionaries
        prediction_rows: List of prediction result dictionaries
        annotated_rgb: Annotated image with bounding boxes
        resized_images: List of (name, image) tuples for resized detections
        final_image: Final image with predictions
    """
    with st.expander("More details"):
        st.subheader("Bounding Box Predictions")
        st.image(annotated_rgb, width="stretch")

        st.subheader("Detections")
        st.dataframe(detections, width="stretch")

        if resized_images:
            st.subheader("Resized Detections")
            for i in range(0, len(resized_images), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(resized_images):
                        _name, img = resized_images[i + j]
                        pred = prediction_rows[i + j]
                        prob = pred["fresh_probability"] if pred["predicted_label"] == "Fresh" else pred["rotten_probability"]
                        caption = f"{pred['predicted_label']} ({prob * 100:.2f}%)"
                        cols[j].image(img, caption=caption, width="stretch")

        if prediction_rows:
            st.subheader("Freshness predictions")
            st.dataframe(prediction_rows, width="stretch")