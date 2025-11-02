# app.py
import streamlit as st
import numpy as np
from PIL import Image
import json
import io
import os
import tensorflow as tf

from tensorflow.keras.applications import vgg16, resnet50, mobilenet_v2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
import requests
from scipy.special import softmax

# ---------------- CONFIG ----------------
# If you already have a model file in the repo, set LOCAL_MODEL_PATH accordingly:
LOCAL_MODEL_PATH = "model/waste_model.h5"   # change if your model file is named differently or in root e.g. "waste_model.h5"

# OR: If model is too big for GitHub, upload to Google Drive and set DRIVE_FILE_ID:
USE_GDRIVE_FALLBACK = True
DRIVE_FILE_ID = ""  # <-- paste your Google Drive file ID here if using remote download

# Classes (from repo README)
DEFAULT_LABELS = ["cardboard", "compost", "glass", "metal", "paper", "plastic", "trash"]
LABELS_JSON = "labels.json"
# ----------------------------------------

st.set_page_config(page_title="Waste / Garbage Classifier", layout="centered")
st.title("♻️ Waste / Garbage Classifier")

# Utility: ensure model exists locally, with optional Google Drive download fallback
def download_from_gdrive(file_id, dest_path):
    if not file_id:
        return False, "No file id provided."
    # Use export=download url
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code != 200:
            return False, f"Download returned status {r.status_code}"
        # For large files, write in chunks
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        return True, "Downloaded"
    except Exception as e:
        return False, str(e)

def ensure_model(local_path=LOCAL_MODEL_PATH):
    if os.path.exists(local_path):
        return True, local_path
    # try root
    root_path = os.path.basename(local_path)
    if os.path.exists(root_path):
        return True, root_path
    # try drive fallback
    if USE_GDRIVE_FALLBACK and DRIVE_FILE_ID:
        st.info("Model not found locally — downloading from Google Drive...")
        ok, msg = download_from_gdrive(DRIVE_FILE_ID, local_path)
        if ok:
            st.success("Model downloaded successfully.")
            return True, local_path
        else:
            st.error(f"Failed to download model: {msg}")
            return False, None
    return False, None

# load labels
def load_labels(path=LABELS_JSON):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                labels = json.load(f)
            if isinstance(labels, list):
                return labels
            if isinstance(labels, dict):
                # convert dict index->label to list (sorted by index key)
                # allow keys as strings.
                max_index = max(int(k) for k in labels.keys())
                lst = [None] * (max_index + 1)
                for k, v in labels.items():
                    lst[int(k)] = v
                return lst
        except Exception as e:
            st.warning(f"Failed to read {path}: {e}")
    # fallback
    return DEFAULT_LABELS

# caching the model so it doesn't reload every interaction
@st.cache_resource
def load_model_cached(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# preprocessing selection
preproc_option = st.sidebar.selectbox("Preprocessing", ["vgg16", "resnet50", "mobilenet_v2", "/255.0"])
show_debug = st.sidebar.checkbox("Show debug outputs", value=False)
labels = load_labels(LABELS_JSON)

# ensure model present
ok, model_file = ensure_model()
if not ok:
    st.warning("Model file not found and could not be downloaded. Put the .h5 file into the repo (model/ or root) or set DRIVE_FILE_ID.")
    st.stop()

# load model
try:
    model = load_model_cached(model_file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# attempt to detect model input size
input_shape = None
try:
    if hasattr(model, "input_shape"):
        input_shape = model.input_shape  # e.g., (None, 224,224,3)
        if isinstance(input_shape, tuple) and len(input_shape) == 4:
            _, h, w, c = input_shape
            target_size = (h or 224, w or 224)
        else:
            target_size = (224, 224)
    else:
        target_size = (224, 224)
except Exception:
    target_size = (224, 224)

st.sidebar.write(f"Model input: {target_size[0]}x{target_size[1]}")

# helper preprocess
def preprocess_image_choice(img: Image.Image, target_size=target_size):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    if preproc_option == "vgg16":
        return vgg_preprocess(arr)
    if preproc_option == "resnet50":
        return resnet50.preprocess_input(arr)
    if preproc_option == "mobilenet_v2":
        return mobilenet_v2.preprocess_input(arr)
    return arr / 255.0

# UI: file uploader + example
uploaded = st.file_uploader("📸 Upload an image (png / jpg / jpeg)", type=["png", "jpg", "jpeg"])
st.write("If you want a quick test and don't have a sample, add sample images to your repo and use the path.")

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read()))
    st.image(image, caption="Uploaded image", use_column_width=True)

    st.write("---")
    st.write("Processing...")

    img_arr = preprocess_image_choice(image, target_size=target_size)
    # run prediction
    try:
        preds = model.predict(img_arr)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # shape normalization
    if preds.ndim == 2 and preds.shape[0] == 1:
        raw = preds[0]
    elif preds.ndim == 1:
        raw = preds
    else:
        raw = preds.flatten()

    # compute softmax probabilities if values look like logits
    try:
        probs = softmax(raw)
    except Exception:
        # fallback: normalize positive
        probs = raw
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)

    top_k = min(5, len(probs))
    top_indices = np.argsort(probs)[-top_k:][::-1]

    result_rows = []
    for idx in top_indices:
        label_name = labels[idx] if idx < len(labels) else f"class_{idx}"
        result_rows.append({"index": int(idx), "label": label_name, "prob": float(probs[idx])})

    st.success(f"Top prediction: {result_rows[0]['label']} ({result_rows[0]['prob']*100:.2f}%)")
    st.table(result_rows)

    if show_debug:
        st.write("Raw model output (first 12):", raw[:12].tolist())
        st.write("Softmax probs (first 12):", probs[:12].tolist())
        st.write("Labels mapping (index -> label):")
        st.write({i: (labels[i] if i < len(labels) else "") for i in range(len(labels))})

else:
    st.info("Upload an image to get predictions. If you prefer, place some sample images in the repo and load them from disk.")
