import streamlit as st
import numpy as np
from PIL import Image
import json
import io
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import requests


# ---------- Auto-download model from Google Drive ----------
MODEL_PATH = "predictWaste12.h5"
DRIVE_FILE_ID = "1SD8B4iRZf8hEnzC7BmNY4WX6fGuGxn4Y"

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        st.info("Model not found locally — downloading from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")
        else:
            st.error("❌ Failed to download model. Please check your Drive link.")
            return False
    return True
# ------------------------------------------------------------


@st.cache_resource
def load_model_cached(path):
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
st.title("♻️ Garbage Classification — Streamlit App")
st.write("Upload a photo of waste and the model will predict its class.")

# Ensure model is available
if ensure_model_exists():
    try:
        model = load_model_cached(MODEL_PATH)
        labels = load_labels("labels.json")

        # Image upload section (this will always show if model loads)
        uploaded = st.file_uploader("📸 Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.read()))
            st.image(image, caption="Uploaded image", use_column_width=True)

            st.write("---")
            st.write("🔄 Processing...")

            img_arr = preprocess_image(image)
            preds = predict(model, img_arr)

            if preds.ndim == 2 and preds.shape[0] == 1:
                preds = preds[0]

            top_idx = int(np.argmax(preds))
            top_conf = float(preds[top_idx])
            label_name = labels[top_idx] if top_idx < len(labels) else str(top_idx)

            st.success(f"Prediction: **{label_name}** ({top_conf * 100:.2f}% confidence)")

            # Top-5 predictions
            top_k = min(5, len(preds))
            top_indices = np.argsort(preds)[-top_k:][::-1]
            rows = [
                {"label": (labels[i] if i < len(labels) else str(i)), "probability": float(preds[i])}
                for i in top_indices
            ]
            st.table(rows)

            # Bar chart
            try:
                import pandas as pd
                df = pd.DataFrame({(labels[i] if i < len(labels) else str(i)): float(preds[i]) for i in range(len(preds))}, index=[0])
                st.bar_chart(df.T)
            except Exception:
                pass

        else:
            st.info("👆 Upload an image to get a prediction.")

    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning("Model could not be loaded. Please check your link or try again later.")
