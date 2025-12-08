import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv

# Image quality validation
import cv2

# Auto scene classifier (MobileNet)
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Nabta AI",
    page_icon="🌿",
    layout="wide"
)


# -------------------------------------------------
# Load scene classifier (automatic detection)
# -------------------------------------------------
scene_model = MobileNetV2(weights="imagenet")


def detect_category(img: Image.Image) -> str:
    arr = img.resize((224, 224))
    arr = np.array(arr).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = scene_model.predict(arr, verbose=0)
    decoded = decode_predictions(preds, top=3)[0]

    labels = [d[1].lower() for d in decoded]
    scores = [d[2] for d in decoded]

    # keywords
    if any(word in labels[0] for word in ["soil", "ground", "earth", "dirt"]):
        return "soil"

    if any(word in labels[0] for word in ["leaf", "plant", "flower", "tree", "vine", "corn", "tomato"]):
        return "plant"

    if scores[0] < 0.45:
        return "unknown"

    return "unknown"


# -------------------------------------------------
# Camera & Image validation
# -------------------------------------------------
def validate_image(img: Image.Image):
    arr = np.array(img)

    # Lighting check
    if arr.mean() < 25:
        st.error("⚠️ Image is too dark. Turn on more light.")
        st.stop()

    # Resolution check
    if arr.shape[0] < 200 or arr.shape[1] < 200:
        st.error("⚠️ Image resolution is too low. Please take a clearer picture.")
        st.stop()

    # Blur check
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_val < 60:
        st.error("⚠️ Image is too blurry. Hold camera steady and retake.")
        st.stop()

    return True


# -------------------------------------------------
# Custom UI Styles
# -------------------------------------------------
# (UNCHANGED — keeping your CSS exactly as before)
# ---- paste your original CSS same as provided ----

st.markdown(""" ... your whole CSS block unchanged ... """, unsafe_allow_html=True)


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🌿 Nabta AI</h1>
        <p>Working towards creating a healthier, greener, and sustainable environment in Kuwait.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# API KEY
# -------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None


# -------------------------------------------------
# Load Keras Models
# -------------------------------------------------
soil_model = None
plant_model = None
soil_model_error = None
plant_model_error = None

try:
    soil_model = keras.models.load_model("models/soil_moisture_model.keras")
except Exception as e:
    soil_model_error = str(e)

try:
    plant_model = keras.models.load_model("models/plant_disease_model.keras")
except Exception as e:
    plant_model_error = str(e)


# -------------------------------------------------
# Labels
# -------------------------------------------------
soil_class_labels = {
    0: "dry",
    1: "moist",
    2: "wet"
}

plant_class_labels = {
    0:"Corn (Cercospora leaf spot - Gray leaf spot)",
    1:"Corn (Common rust)",
    2:"Corn (Northern Leaf Blight)",
    3:"Corn (Healthy)",
    4:"Pepper (Bacterial spot)",
    5:"Pepper (Healthy)",
    6:"Potato (Early blight)",
    7:"Potato (Late blight)",
    8:"Potato (Healthy)",
    10:"Strawberry (Leaf scorch)",
    11:"Strawberry (Healthy)",
    12:"Tomato (Bacterial spot)",
    13:"Tomato (Early blight)",
    14:"Tomato (Late blight)",
    15:"Tomato (Leaf Mold)",
    16:"Tomato (Septoria leaf spot)",
    17:"Tomato (Spider mites / Two-spotted spider mite)",
    18:"Tomato (Target Spot)",
    19:"Tomato (Yellow Leaf Curl Virus)",
    20:"Tomato (Mosaic virus)",
    21:"Tomato (Healthy)"
}


# -------------------------------------------------
# Image Preprocessing
# -------------------------------------------------
def preprocess_image(img: Image.Image, target_size=(150, 150)):
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# -------------------------------------------------
# Prediction logic with confidence filtering
# -------------------------------------------------
def predict_soil(img: Image.Image):
    if soil_model is None:
        return f"[Soil model not loaded: {soil_model_error}]", 0.0
    preds = soil_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    label = soil_class_labels.get(idx, "Unknown")
    return label, prob


def predict_plant(img: Image.Image):
    if plant_model is None:
        return f"[Plant model not loaded: {plant_model_error}]", 0.0
    preds = plant_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    label = plant_class_labels.get(idx, "Unknown")
    return label, prob


# -------------------------------------------------
# Gemini advice (unchanged)
# -------------------------------------------------
def explain_prediction(label: str, category: str) -> str:
    # (unchanged - paste your existing block)
    # keeps full dual language instructions
    ...


# -------------------------------------------------
# Layout
# -------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    ...
    # keep everything exactly as it was (Upload/Camera UI)
    ...


with right_col:
    ...
    # preview
    ...


st.markdown("---")


# -------------------------------------------------
# Analyze button and NEW automatic detection
# -------------------------------------------------





# -------------------------------------------------
# Run prediction
# -------------------------------------------------
if analyze_clicked:

    if task_type == "Soil Moisture":
        label, prob = predict_soil(img)
        if prob < 0.60:
            st.error("⚠️ This does not seem to be soil. Try another picture.")
            st.stop()
        explanation_raw = explain_prediction(label, "soil moisture")

    else:
        label, prob = predict_plant(img)
        if prob < 0.60:
            st.error("⚠️ This does not seem to be a plant. Try another picture.")
            st.stop()
        explanation_raw = explain_prediction(label, "plant disease")


    # everything else stays EXACTLY as your original rendering:
    # result card + English + Arabic blocks
    ...



