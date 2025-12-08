import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------------
# EXTRA imports for scene detection
# -------------------------------------
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

# -------------------------------------
# Extra import for blur detection
# -------------------------------------
import cv2

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

    if any(word in labels[0] for word in [
        "leaf", "plant", "flower", "tree",
        "vine", "corn", "tomato"
    ]):
        return "plant"

    if scores[0] < 0.45:
        return "unknown"

    return "unknown"


# -------------------------------------------------
# Image validation (blur, lighting)
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
# Environment
# -------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None


# -------------------------------------------------
# Load ML Models
# -------------------------------------------------
soil_model = None
plant_model = None

try:
    soil_model = keras.models.load_model("models/soil_moisture_model.keras")
except:
    soil_model = None

try:
    plant_model = keras.models.load_model("models/plant_disease_model.keras")
except:
    plant_model = None


# -------------------------------------------------
# Labels
# -------------------------------------------------
soil_class_labels = {0:"dry", 1:"moist", 2:"wet"}

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
# Predict functions
# -------------------------------------------------
def preprocess_image(img: Image.Image, size=(150,150)):
    img = img.resize(size)
    arr = np.array(img).astype("float32")/255.0
    return np.expand_dims(arr,0)

def predict_soil(img):
    preds = soil_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    return soil_class_labels[idx], float(preds[0][idx])

def predict_plant(img):
    preds = plant_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    return plant_class_labels[idx], float(preds[0][idx])


# -------------------------------------------------
# Explain prediction
# -------------------------------------------------
def explain_prediction(label, category):
    if not gemini_model:
        return "Gemini not configured."
    prompt = f"""
Explain {category}: {label}
in English and Arabic
with steps and next actions.
"""
    try:
        return gemini_model.generate_content(prompt).text
    except:
        return "Gemini error."


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🌿 Nabta AI")

img = None

choice = st.radio("Image input:", ["Upload", "Camera"])

if choice=="Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
else:
    cam = st.camera_input("Take a photo")
    if cam:
        img = Image.open(cam).convert("RGB")

task_type = st.radio("Task:", ["Soil Moisture","Plant Disease"])

if img:
    st.image(img,use_column_width=True)

if st.button("Analyze") and img:

    validate_image(img)

    category = detect_category(img)

    if task_type=="Soil Moisture" and category!="soil":
        st.error("⚠️ This image does not appear to be soil.")
        st.stop()

    if task_type=="Plant Disease" and category!="plant":
        st.error("⚠️ This image does not appear to be a plant leaf.")
        st.stop()

    if task_type=="Soil Moisture":
        label,prob = predict_soil(img)
    else:
        label,prob = predict_plant(img)

    st.success(f"Prediction: {label} (confidence {prob:.2f})")

    st.write(explain_prediction(label,task_type.lower()))
