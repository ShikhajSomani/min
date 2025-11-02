import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
import requests

# ---------------- Config ----------------
MODEL_PATH = "predictWaste12.h5"
DRIVE_FILE_ID = "1SD8B4iRZf8hEnzC7BmNY4WX6fGuGxn4Y"

LABELS = [
    "cardboard",
    "compost",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]
# ----------------------------------------

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        st.info("Model not found locally — downloading from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        r = requests.get(url, allow_redirects=True)
        open(MODEL_PATH, 'wb').write(r.content)
        st.success("✅ Model downloaded successfully!")
    return True


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0  # simple normalization
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, img_array):
    preds = model.predict(img_array)
    preds = tf.nn.softmax(preds).numpy()[0]
    return preds


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("♻ Garbage Classification — Streamlit App")
st.write("Upload a garbage image, and the model will classify it into one of 7 categories.")

if ensure_model_exists():
    try:
        model = load_model()
        st.sidebar.write("✅ Model loaded successfully.")

        uploaded = st.file_uploader("📸 Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded:
            image = Image.open(io.BytesIO(uploaded.read()))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.write("🔄 Predicting...")
            img_array = preprocess_image(image)
            preds = predict(model, img_array)

            top_idx = int(np.argmax(preds))
            st.success(f"### Prediction: {LABELS[top_idx]}")
            st.write(f"Confidence: {preds[top_idx]*100:.2f}%")

            # Probabilities table
            import pandas as pd
            df = pd.DataFrame({
                "Class": LABELS,
                "Probability": [float(p) for p in preds]
            }).sort_values("Probability", ascending=False)
            st.bar_chart(df.set_index("Class"))

        else:
            st.info("👆 Upload an image to start classification.")

    except Exception as e:
        st.error(f"Error: {e}")
