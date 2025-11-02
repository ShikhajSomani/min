import streamlit as st
import numpy as np
from PIL import Image
import json
import io
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import requests  # ✅ use requests instead of gdown


# ---------- Auto-download model from Google Drive ----------
MODEL_PATH = "predictWaste12.h5"
DRIVE_FILE_ID = "1SD8B4iRZf8hEnzC7BmNY4WX6fGuGxn4Y"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        # Use the official export=download URL
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error(f"Failed to download model. Status code: {response.status_code}")
            st.stop()
# ------------------------------------------------------------


@st.cache_resource
def load_model(path=MODEL_PATH):
    return tf.keras.models.load_model(path)


def load_labels(path="labels.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            labels = json.load(f)
            if isinstance(labels, dict):
                labels = [labels[str(i)] if str(i) in labels else labels[i] for i in range(len(labels))]
            return labels
    # fallback list
    return [
        "class_0", "class_1", "class_2", "class_3", "class_4",
        "class_5", "class_6", "class_7", "class_8", "class_9",
        "class_10", "class_11",
    ]


def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict(model, img_array):
    preds = model.predict(img_array)
    return preds


# Streamlit UI
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("Garbage Classification — Streamlit App")
st.write("Upload a photo of waste and the model will predict its class.")

model_path = MODEL_PATH
labels_path = st.sidebar.text_input("Labels path (json)", value="labels.json")

if not os.path.exists(model_path):
    st.sidebar.error(f"Model file not found at: {model_path}. Please check path or upload model.")
else:
    model = load_model(model_path)
    labels = load_labels_
