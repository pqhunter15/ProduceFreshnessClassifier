import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import cv2
import tempfile
import os

st.set_page_config(page_title="Banana Detector", layout="wide")
st.title("Banana Detector")
st.write("Take a picture or upload an image, then run YOLO detection.")

MODEL_PATH = "weights/best4-all-heavy.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

source_option = st.radio(
    "Choose image source",
    ["Take a picture", "Upload an image"],
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

conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

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

    with col2:
        st.subheader("Prediction")
        st.image(annotated_rgb, width="stretch")

    st.subheader("Detections")
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

        st.dataframe(detections, width="stretch")

    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass