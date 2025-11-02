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
MODEL_PATH = "predictWaste12.h5"  # rename if your file has a different name
DRIVE_FILE_ID = "1SD8B4iRZf8hEnzC7BmNY4WX6fGuGxn4Y"

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        st.info("Model not found locally — downloading from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        if 'confirm=t' in response.url:
            confirm_token = response.url.split('confirm=')[1].split('&')[0]
            url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm={confirm_token}"
            response = session.get(url, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024 * 1024  # 1MB
            progress_bar = st.progress(0)
            bytes_downloaded = 0

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size > 0:
                            percent_complete = int(100 * bytes_downloaded / total_size)
                            progress_bar.progress(percent_complete)
            
            progress_bar.empty()
            st.success("✅ Model downloaded successfully!")
        else:
            st.error(f"❌ Failed to download model. Status: {response.status_code}")
            return False
    return True
# ------------------------------------------------------------


@st.cache_resource
def load_model_cached(path):
    return tf.keras.models.load_model(path, compile=False)


def load_labels():
    """Fixed label list for 7-class garbage classification model."""
    return [
        "cardboard",
        "compost",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash"
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


# ---------------- Streamlit UI -----------------
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("♻ Garbage Classification — Streamlit App")
st.write("Upload a waste image and the model will classify it into one of 7 categories.")

if ensure_model_exists():
    try:
        model = load_model_cached(MODEL_PATH)
        labels = load_labels()

        # check if model output matches label count
        num_classes_model = model.output_shape[-1]
        if len(labels) != num_classes_model:
            st.warning(f"⚠ Model expects {num_classes_model} outputs but {len(labels)} labels loaded. "
                       "Predictions may not align perfectly.")

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

            preds = tf.nn.softmax(preds).numpy()
            top_idx = int(np.argmax(preds))
            top_conf = float(preds[top_idx])
            label_name = labels[top_idx] if top_idx < len(labels) else f"Class {top_idx}"

            st.success(f"### Prediction: {label_name}")
            st.subheader(f"Confidence: {top_conf * 100:.2f}%")

            # show probabilities
            import pandas as pd
            st.markdown("#### Class Probabilities")
            df = pd.DataFrame({
                "Class": labels,
                "Probability": [float(p) for p in preds]
            })
            df = df.sort_values("Probability", ascending=False)
            st.bar_chart(df.set_index("Class"))

        else:
            st.info("👆 Upload an image to start classification.")

    except Exception as e:
        st.error(f"Error running prediction: {e}")
else:
    st.error("❌ Model file missing and could not be downloaded.")
