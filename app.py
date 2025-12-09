import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Nabta AI",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------------------------
# Custom UI Styles
# -------------------------------------------------
st.markdown(
    """
    <style>
    ... EXACTLY SAME CSS ...
    </style>
    """,
    unsafe_allow_html=True
)

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
# Environment / API Key
# -------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None

# -------------------------------------------------
# Load Models (safe fallback)
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
soil_class_labels = { ... }
plant_class_labels = { ... }

# -------------------------------------------------
# Image Preprocessing
# -------------------------------------------------
def preprocess_image(img: Image.Image, target_size=(150, 150)):
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------------------------------------
# Prediction logic
# -------------------------------------------------
def predict_soil(img: Image.Image):
    ...
def predict_plant(img: Image.Image):
    ...

# -------------------------------------------------
# Gemini Advice (English + Arabic)
# -------------------------------------------------
def explain_prediction(label: str, category: str) -> str:
    if not gemini_model:
        return (
            "🌐 Gemini is not configured. Add your GEMINI_API_KEY in Streamlit Secrets."
        )

    prompt = (
        f"You are an experienced agricultural field advisor who helps farmers in real conditions. "
        f"The AI system predicted {category} = \"{label}\".\n\n"
        f"Your job:\n"
        f"1. Explain what this result means and why it matters.\n"
        f"2. Give clear, practical next steps the farmer should take in the next 24 hours.\n"
        f"3. Give prevention tips for the next few days.\n"
        f"4. If it is a disease, explain if the crop should be isolated, sprayed, pruned, or monitored.\n"
        f"5. If it is soil moisture, give watering guidance: how much, how often, and what to watch for.\n\n"
        f"Answer in TWO sections:\n\n"
        f"### English Explanation\n"
        f"- Write in simple English for a non-technical farmer.\n"
        f"- Use bullet points for actions.\n\n"
        f"### Arabic Explanation (الفهم بالعربية)\n"
        f"- اكتب شرحاً تفصيلياً باللغة العربية الفصحى السهلة.\n"
        f"- استخدم نقاط واضحة لخطوات العمل.\n"
        f"- اجعل النص عملي جداً (مثل: اسقِ التربة الآن / افحص الأوراق غداً / اعزل النبتة إذا كانت مصابة).\n"
    )

    try:
        resp = gemini_model.generate_content(prompt)

        # 🔥 NEW SAFE GEMINI TEXT EXTRACTION
        if resp.candidates and resp.candidates[0].content.parts:
            text = resp.candidates[0].content.parts[0].text
            return text.strip() if text else "No explanation generated."

        return "No explanation generated (empty Gemini response)."

    except Exception as e:
        return f"Gemini explanation unavailable right now: {e}"

# -------------------------------------------------
# Layout: Input / Preview Columns
# -------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

... EVERYTHING BELOW REMAINS EXACTLY THE SAME ...

