import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

# Load label mappings
with open("labels.json", "r") as f:
    labels = json.load(f)

st.title("♻️ Waste or Garbage Classification")
st.write("Upload an image of waste material and let the model classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Classifying...")
    model = load_model()

    # Preprocess the image
    img = image.resize((224, 224))  # adjust if your model expects different input size
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if model trained that way

    # Predict
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    pred_label = labels[str(pred_index)]
    confidence = np.max(predictions[0]) * 100

    # Show result
    st.markdown(f"### 🏷️ Prediction: **{pred_label.capitalize()}**")
    st.markdown(f"### 🔢 Confidence: **{confidence:.2f}%**")
