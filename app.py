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
            confirm_token = response.url.split('c_
