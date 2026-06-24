import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

model = YOLO("best.pt")

st.title("Parasite Image Classifier")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        results = model(tmp.name)

        plotted = results[0].plot()

        st.image(plotted)
