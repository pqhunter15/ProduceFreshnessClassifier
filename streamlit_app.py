import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2
import tempfile
import os

from src.freshness_inference.model import FreshnessClassifier

st.set_page_config(page_title="Banana Detector", layout="wide")
st.title("Banana Detector")
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

model_group = st.radio(
    "Select model group",
    list(MODEL_GROUPS.keys()),
    index=0,
    horizontal=True,
)
model_version = st.radio(
    "Select version",
    ["Light", "Heavy"],
    index=0,
    horizontal=True,
)
selected_model_conf = MODEL_GROUPS[model_group][model_version]
st.caption(selected_model_conf["details"])

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def load_freshness_classifier():
    return FreshnessClassifier()

model = load_model(selected_model_conf["path"])
classifier = load_freshness_classifier()

source_option = st.radio(
    "Choose image source",
    ["Upload an image", "Take a picture"],
    horizontal=True
)

image_file = None

if source_option == "Take a picture":
    image_file = st.camera_input("Take a banana picture")
else:
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

conf = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.05)

if image_file is not None:
    # Open image and fix hidden phone/camera orientation metadata
    pil_img = Image.open(image_file)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    pil_img = pil_img.resize((1280, 720))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original image")
        st.image(pil_img, width="stretch")

    # Save uploaded/captured image to a temporary file
    # so YOLO reads it the same way it would from disk in Colab
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        temp_path = tmp_file.name
        pil_img.save(temp_path, format="JPEG")

    with st.spinner("Running detection..."):
        results = model.predict(
            source=temp_path,
            conf=conf,
            verbose=False
        )
        r = results[0]

        annotated_bgr = r.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if r.boxes is None or len(r.boxes) == 0:
        st.write("No detections found.")
    else:
        detections = []
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class": model.names[cls_id],
                "confidence": round(conf_score, 3),
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1)
            })

        # Target dimension (Width, Height)
        target_size = (224, 224)

        resized_images = []
        for detection in detections:
            name = detection['class']
            xmin, ymin, xmax, ymax = detection['x1'], detection['y1'], detection['x2'], detection['y2']

            # Convert float coordinates to integers for slicing
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # Step 1: Extract (Crop)
            # Corrected slicing order: [ymin:ymax, xmin:xmax]
            crop = annotated_rgb[ymin:ymax, xmin:xmax]
            # Step 2: Resize to 224x224
            # Note: cv2.resize takes (width, height)
            resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            resized_images.append((name, resized))

        # Predict freshness from the resized images
        prediction_rows = []
        for idx, (detection_name, img) in enumerate(resized_images):
            prediction = classifier.predict_array(img, image_path=f"resized_{idx}")
            row = prediction.to_dict()
            row["detection_name"] = detection_name
            row["fresh_probability"] = round(row["fresh_probability"], 2)
            row["rotten_probability"] = round(row["rotten_probability"], 2)
            prediction_rows.append(row)

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

        rotten_preds = [p for p in prediction_rows if p["predicted_label"] == "Rotten"]
        if rotten_preds:
            final_pred = max(rotten_preds, key=lambda p: p["rotten_probability"])
            final_label = "Rotten"
            final_prob = final_pred["rotten_probability"]
        else:
            final_pred = max(prediction_rows, key=lambda p: p["fresh_probability"])
            final_label = "Fresh"
            final_prob = final_pred["fresh_probability"]

        final_caption = f"Final result: {final_label} ({final_prob * 100:.2f}%)"
        with col2:
            st.subheader("Prediction")

            # Draw bounding boxes and predicted label/probability on the final image
            final_image = pil_img.copy()
            draw = ImageDraw.Draw(final_image)
            font = ImageFont.load_default()

            for idx, detection in enumerate(detections):
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

            st.image(final_image, width="stretch")



    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass