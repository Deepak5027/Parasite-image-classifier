import os
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Parasite Detector", layout="centered")

st.title("🦠 Parasite Image Detection System")

# ---------------------------
# LOAD MODEL SAFELY
# ---------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")

    if not os.path.exists(model_path):
        st.error("❌ best.pt not found in project directory")
        return None

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error("❌ Failed to load model. Your best.pt may be corrupted or incompatible.")
        st.exception(e)
        return None


model = load_model()

# ---------------------------
# CONFIDENCE SLIDER
# ---------------------------
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# ---------------------------
# IMAGE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, channels="BGR", caption="Original Image")

    # ---------------------------
    # RUN DETECTION
    # ---------------------------
    results = model.predict(image, conf=conf_threshold, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    st.subheader("Result")
    st.image(image, channels="BGR")
