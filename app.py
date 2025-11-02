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
        # Use the requests library to download the file
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        
        # We need to handle Google's cookie/redirect for large files
        # First request to get the redirect and cookies
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check for confirmation form (typical for large files)
        if 'confirm=t' in response.url:
            # Extract the confirmation token
            confirm_token = response.url.split('confirm=')[1].split('&')[0]
            url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm={confirm_token}"
            response = session.get(url, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024 * 1024 # 1 MB chunks
            
            progress_bar = st.progress(0)
            bytes_downloaded = 0

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        # Update progress bar
                        if total_size > 0:
                            percent_complete = int(100 * bytes_downloaded / total_size)
                            progress_bar.progress(percent_complete)
            
            progress_bar.empty()
            st.success("✅ Model downloaded successfully!")
        else:
            st.error(f"❌ Failed to download model. Status Code: {response.status_code}. Please check your Drive link or permissions.")
            return False
    return True
# ------------------------------------------------------------


@st.cache_resource
def load_model_cached(path):
    # Set compile=False for safety when loading models in different environments
    return tf.keras.models.load_model(path, compile=False)


def load_labels(path="labels.json"):
    # --- FIXED: Robust and descriptive fallback labels ---
    # Assuming a 12-class model structure common in waste classification datasets
    default_labels = [
        "Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash/General Waste",
        "Battery", "Biological/Organic", "Clothes/Textile", "E-waste", "Shoes/Leather", "White Glass"
    ]
    
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                labels_data = json.load(f)
                
                if isinstance(labels_data, list):
                    # Simple list of labels
                    return labels_data
                elif isinstance(labels_data, dict):
                    # Handle dictionary like {"0": "Cardboard", "1": "Glass", ...}
                    # Ensure keys are integers for correct sorting
                    keys = [int(k) for k in labels_data.keys()]
                    if keys:
                        # Sort dictionary keys numerically to ensure correct index order
                        sorted_labels = [labels_data[str(i)] for i in sorted(keys)]
                        return sorted_labels
        except Exception as e:
            st.warning(f"Could not load or parse '{path}' due to: {e}. Falling back to default labels.")
            
    # Return default labels if file is missing or unreadable
    return default_labels


def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Preprocesses the image for VGG-based models."""
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    # Use VGG16-specific preprocessing (subtracts ImageNet means)
    arr = preprocess_input(arr)
    return arr


def predict(model, img_array):
    """Generates prediction and handles softmax if needed."""
    preds = model.predict(img_array)
    return preds


# Streamlit UI
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("♻ Garbage Classification — Streamlit App")
st.write("Upload a photo of waste and the model will predict its class.")

# Ensure model is available
if ensure_model_exists():
    try:
        model = load_model_cached(MODEL_PATH)
        labels = load_labels("labels.json")
        
        # Check if the number of labels matches the model's output size
        num_classes_model = model.output_shape[-1]
        if len(labels) != num_classes_model:
            st.error(f"⚠ Label mismatch! The model expects {num_classes_model} classes, but {len(labels)} labels were loaded. Predictions might be mislabeled.")
            if len(labels) < num_classes_model:
                 st.warning("If you are missing the labels.json file, the fallback labels may not align with your model's training data.")


        # Image upload section (this will always show if model loads)
        uploaded = st.file_uploader("📸 Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.read()))
            st.image(image, caption="Uploaded image", use_column_width=True)

            st.markdown("---")
            st.write("### 🔄 Running Prediction...")

            img_arr = preprocess_image(image)
            preds = predict(model, img_arr)

            # Ensure preds is a 1D array of probabilities
            if preds.ndim == 2 and preds.shape[0] == 1:
                preds = preds[0]
                
            # Use softmax if the model output is not already normalized probabilities
            # Keras classification models usually output logits or probabilities, let's ensure it's probabilities
            if np.sum(preds) < 0.95 or np.sum(preds) > 1.05:
                preds = tf.nn.softmax(preds).numpy()

            top_idx = int(np.argmax(preds))
            top_conf = float(preds[top_idx])
            
            # Safely get the label name
            label_name = labels[top_idx] if top_idx < len(labels) else f"Unknown Class Index {top_idx}"

            st.success(f"### Prediction: *{label_name}*")
            st.subheader(f"Confidence: {top_conf * 100:.2f}%")


            # Top-5 predictions
            st.markdown("---")
            st.write("#### Top 5 Probabilities")
            top_k = min(5, len(preds))
            # Get indices of the top predictions
            top_indices = np.argsort(preds)[-top_k:][::-1]
            
            rows = []
            for i in top_indices:
                rows.append({
                    "Category": labels[i] if i < len(labels) else f"Unknown Index {i}", 
                    "Probability": float(preds[i])
                })
            
            import pandas as pd
            df_table = pd.DataFrame(rows)
            st.dataframe(df_table, hide_index=True, use_container_width=True)


            # Bar chart
            st.write("#### All Class Probabilities")
            try:
                # Create a DataFrame for the bar chart visualization
                chart_data = {
                    "Class": [(labels[i] if i < len(labels) else f"Index {i}") for i in range(len(preds))],
                    "Probability": [float(preds[i]) for i in range(len(preds))]
                }
                df_chart = pd.DataFrame(chart_data)
                df_chart = df_chart.set_index('Class')

                st.bar_chart(df_chart)
            except Exception:
                pass

        else:
            st.info("👆 Upload an image to start the classification process.")

    except Exception as e:
        st.error(f"Error initializing or running prediction: {e}")
        st.caption("Please ensure your Keras model is a valid classification model compatible with VGG16 preprocessing.")
else:
    st.warning("Model could not be loaded. Please ensure the Google Drive file ID is correct and accessible.")
