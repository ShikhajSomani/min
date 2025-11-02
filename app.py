import streamlit as st
import numpy as np
from PIL import Image
import json
import io
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


@st.cache_resource
def load_model(path="predictWaste12.h5"):
    return tf.keras.models.load_model(path)


def load_labels(path="labels.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            labels = json.load(f)
            if isinstance(labels, dict):
                # allow either list or dict mapping index->label
                # if dict, sort by key to make a list
                labels = [labels[str(i)] if str(i) in labels else labels[i] for i in range(len(labels))]
            return labels
    # fallback: make the user edit this list to match model training order
    return [
        "class_0",
        "class_1",
        "class_2",
        "class_3",
        "class_4",
        "class_5",
        "class_6",
        "class_7",
        "class_8",
        "class_9",
        "class_10",
        "class_11",
    ]


def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # VGG16 preprocessing
    return arr


def predict(model, img_array):
    preds = model.predict(img_array)
    return preds


# Streamlit UI
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("Garbage Classification — Streamlit App")
st.write("Upload a photo of waste and the model will predict its class.")

model_path = st.sidebar.text_input("Model path", value="predictWaste12.h5")
labels_path = st.sidebar.text_input("Labels path (json)", value="labels.json")

if not os.path.exists(model_path):
    st.sidebar.error(f"Model file not found at: {model_path}. Please check path or upload model.")
else:
    model = load_model(model_path)
    labels = load_labels(labels_path)

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Uploaded image", use_column_width=True)

        st.write("---")
        st.write("Processing...")

        img_arr = preprocess_image(image, target_size=(224, 224))
        preds = predict(model, img_arr)

        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]

        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx])
        label_name = labels[top_idx] if top_idx < len(labels) else str(top_idx)

        st.success(f"Prediction: {label_name} ({top_conf * 100:.2f}% confidence)")

        # Display top-5 predictions
        top_k = min(5, len(preds))
        top_indices = np.argsort(preds)[-top_k:][::-1]
        rows = [
            {"label": (labels[i] if i < len(labels) else str(i)), "probability": float(preds[i])}
            for i in top_indices
        ]
        st.table(rows)

        # Optional bar chart
        try:
            import pandas as pd
            df = pd.DataFrame({(labels[i] if i < len(labels) else str(i)): float(preds[i]) for i in range(len(preds))}, index=[0])
            st.bar_chart(df.T)
        except Exception:
            pass
    else:
        st.info("Upload an image to get a prediction.")
