import streamlit as st
import numpy as np
from PIL import Image
import json
import io
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import requests
import pandas as pd

# ----------- CONFIGURATION -----------
MODEL_PATH = "predictWaste12.h5"  # Model filename
DRIVE_FILE_ID = "1SD8B4iRZf8hEnzC7BmNY4WX6fGuGxn4Y"  # Replace if you upload a new model
LABELS = [
    "cardboard",
    "compost",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]
# ------------------------------------


# ---------- Download model if missing ----------
def ensure_model_exists():
    """Download model file from Google Drive if not found locally."""
    if not os.path.exists(MODEL_PATH):
        st.info("Model not found locally — downloading from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

        session = requests.Session()
        response = session.get(url, stream=True)

        if response.status_code != 200:
            st.error(f"❌ Failed to download model. Status: {response.status_code}")
            return False

        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 1024  # 1MB chunks
        progress_bar = st.progress(0)
        bytes_downloaded = 0

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(100 * bytes_downloaded / total_size)
                        progress_bar.progress(percent)

        progress_bar.empty()
        st.success("✅ Model downloaded successfully!")
    return True


# ---------- Load model ----------
@st.cache_resource
def load_model_cached(path):
    return tf.keras.models.load_model(path, compile=False)


# ---------- Image preprocessing ----------
def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


# ---------- Prediction ----------
def predict(model, img_array):
    preds = model.predict(img_array)
    return preds


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("♻ Garbage Classification — Streamlit App")
st.write("Upload a photo of waste and the model will predict its class.")


if ensure_model_exists():
    try:
        model = load_model_cached(MODEL_PATH)

        uploaded = st.file_uploader("📸 Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.read()))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.markdown("---")
            st.write("### 🔄 Running Prediction...")

            img_arr = preprocess_image(image)
            preds = predict(model, img_arr)[0]

            # Convert logits to probabilities (softmax)
            preds = tf.nn.softmax(preds).numpy()

            top_idx = int(np.argmax(preds))
            top_conf = float(preds[top_idx])
            label_name = LABELS[top_idx] if top_idx < len(LABELS) else f"Unknown class {top_idx}"

            # --- Display result nicely ---
            st.write("### 🏷 Prediction Result")

            if top_conf < 0.4:
                st.error(f"⚠ The model is unsure ({top_conf*100:.2f}% confidence).")
                st.write(f"Still, it *thinks this might be* **{label_name}**.")
                st.caption("This might happen if lighting, background, or angle affects the image.")
            elif top_conf < 0.7:
                st.warning(f"🤔 Possible match: **{label_name}** ({top_conf*100:.2f}% confidence)")
                st.caption("Confidence is moderate. Try uploading a clearer photo.")
            else:
                st.success(f"### ✅ Predicted: *{label_name}*")
                st.subheader(f"Confidence: {top_conf*100:.2f}%")

            # --- Top-5 predictions table ---
            st.markdown("---")
            st.write("#### Top Probabilities")
            top_indices = np.argsort(preds)[::-1][:5]

            df_table = pd.DataFrame({
                "Category": [LABELS[i] for i in top_indices],
                "Probability (%)": [round(preds[i]*100, 2) for i in top_indices]
            })
            st.dataframe(df_table, hide_index=True, use_container_width=True)

            # --- Bar chart for all classes ---
            st.write("#### Class Probability Distribution")
            df_chart = pd.DataFrame({
                "Class": LABELS,
                "Probability": preds
            }).set_index("Class")
            st.bar_chart(df_chart)

        else:
            st.info("👆 Upload an image to start classification.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.warning("Model could not be loaded. Please verify your Google Drive link.")
