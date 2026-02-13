import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv
from db import create_table
from auth import create_user, login_user, get_all_users, delete_user

# Page Configuration
st.set_page_config(
    page_title="Nabta AI",
    page_icon="🌿",
    layout="wide"
)

create_table()

# Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# REGISTER 
if choice == "Register":
    st.subheader("Create Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if create_user(username, password, role="user"):
            st.success("Account created successfully!")
        else:
            st.error("Username already exists.")

# LOGIN 
elif choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        role = login_user(username, password)
        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.success(f"Logged in as {role}")
        else:
            st.error("Invalid credentials")

# Security
if st.session_state.logged_in:

    st.sidebar.success(f"Role: {st.session_state.role}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.experimental_rerun()

    # ADMIN Page 
    if st.session_state.role == "admin":
        st.title("Admin Dashboard")

        st.subheader("All Users")
        users = get_all_users()

        for user in users:
            user_id, username, role = user
            col1, col2, col3 = st.columns([3,2,1])
            col1.write(username)
            col2.write(role)

            if role != "admin":  # Prevent deleting admin
                if col3.button("Delete", key=user_id):
                    delete_user(user_id)
                    st.experimental_rerun()

    # USER Page 
    elif st.session_state.role == "user":
        st.title("User Dashboard")
        st.write("Welcome to your application 🎉")

# Custom UI Styles
st.markdown(
    """
    <style>
    body {
        background-color: #f7f8fa;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
    }
    .main-header {
        background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
        color: #fff;
    }
    .main-header p {
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
        color: rgba(255,255,255,0.9);
    }

    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 24px rgba(0,0,0,0.04);
    }

    .card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #2e7d32;
        margin-top: 0;
        margin-bottom: .75rem;
        display: flex;
        align-items: center;
        gap: .5rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: .5rem;
    }

    /* Result card where we show prediction + confidence */
    .result-card {
        background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 100%);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        border: none;
        box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #ffffff;
    }

    .result-label {
        font-weight: 600;
        font-size: 1.2rem;
        color: #ffffff;
        margin-bottom: .25rem;
    }

    .confidence {
        font-size: 1rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 0.75rem;
    }

    /* White advice box for English */
    .advice-wrapper {
        background: #ffffff;
        border: 2px solid #2e7d32;
        border-radius: 10px;
        padding: 1rem 1rem;
        font-size: 0.9rem;
        color: #1f2937;
        line-height: 1.5rem;
        white-space: pre-wrap;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    .advice-header {
        font-weight: 600;
        margin-bottom: .5rem;
        font-size: .95rem;
        color: #374151;
    }

    /* Arabic block, RTL */
    .rtl-block {
        direction: rtl;
        text-align: right;
        background: #ffffff;
        border-radius: 10px;
        border: 2px solid #2e7d32;
        padding: .9rem .9rem;
        margin-top: .75rem;
        font-size: 0.9rem;
        line-height: 1.6rem;
        color: #1f2937;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05);
    }

    /* Big green call-to-action button */
    .analyze-button button {
        width: 100% !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 100%) !important;
        border: 0 !important;
        color: white !important;
        box-shadow: 0 10px 24px rgba(46,125,50,.4) !important;
    }

    /* Warning box */
    .warning-box {
        background: #fff7ed;
        border: 1px solid #fdba74;
        color: #9a3412;
        border-radius: 10px;
        padding: .8rem 1rem;
        font-size: .9rem;
    }
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
    # If this model gives 404 in logs, switch to "gemini-pro-latest"
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
soil_class_labels = {
    0: "dry",
    1: "moist",
    2: "wet"
}

plant_class_labels = {
    0: "Corn (Cercospora leaf spot - Gray leaf spot)",
    1: "Corn (Common rust)",
    2: "Corn (Northern Leaf Blight)",
    3: "Corn (Healthy)",
    4: "Pepper (Bacterial spot)",
    5: "Pepper (Healthy)",
    6: "Potato (Early blight)",
    7: "Potato (Late blight)",
    8: "Potato (Healthy)",
    10: "Strawberry (Leaf scorch)",
    11: "Strawberry (Healthy)",
    12: "Tomato (Bacterial spot)",
    13: "Tomato (Early blight)",
    14: "Tomato (Late blight)",
    15: "Tomato (Leaf Mold)",
    16: "Tomato (Septoria leaf spot)",
    17: "Tomato (Spider mites / Two-spotted spider mite)",
    18: "Tomato (Target Spot)",
    19: "Tomato (Yellow Leaf Curl Virus)",
    20: "Tomato (Mosaic virus)",
    21: "Tomato (Healthy)"
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
# Prediction logic
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

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>📥 Input Image</h3>', unsafe_allow_html=True)

    input_method = st.radio(
        "Choose how to provide an image:",
        ("Upload Image", "Use Camera")
    )

    img = None
    if input_method == "Upload Image":
        uploaded = st.file_uploader(
            "Upload a soil or plant image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
    else:
        cam_img = st.camera_input("Take a live photo")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Preview & Task</h3>', unsafe_allow_html=True)

    if img is not None:
        st.image(img, caption="Preview", use_container_width=True)
    else:
        st.markdown(
            '<div class="warning-box">No image yet. Upload or take a photo.</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">What do you want to analyze?</div>', unsafe_allow_html=True)
    task_type = st.radio(
        "",
        ["Soil Moisture", "Plant Disease"],
        horizontal=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# Analyze button row
# -------------------------------------------------
analyze_clicked = False
if img is not None:
    with st.container():
        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Image with Nabta")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="warning-box">Please provide an image first to run analysis.</div>',
        unsafe_allow_html=True
    )

# -------------------------------------------------
# Results Section
# -------------------------------------------------
if analyze_clicked and img is not None:
    with st.spinner("Analyzing image and generating advice..."):
        if task_type == "Soil Moisture":
            label, prob = predict_soil(img)
            explanation_raw = explain_prediction(label, "soil moisture")
        else:
            label, prob = predict_plant(img)
            explanation_raw = explain_prediction(label, "plant disease")

    # Split English / Arabic for nicer layout
    english_part = ""
    arabic_part = ""

    if "### Arabic Explanation" in explanation_raw:
        parts = explanation_raw.split("### Arabic Explanation")
        english_part = parts[0].replace("### English Explanation", "").strip()
        arabic_part = parts[1].strip()
    else:
        english_part = explanation_raw

    # ✅ White text prediction card (inline style so Streamlit doesn't override)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                ✅ Prediction:
                <span style="color:#ffffff;">{label}</span>
            </div>
            <div class="confidence">
                Confidence: {prob:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # English advice box
    st.markdown('<div class="advice-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="advice-header">English Guidance</div>', unsafe_allow_html=True)
    st.markdown(english_part, unsafe_allow_html=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # Arabic advice box
    if arabic_part:
        st.markdown('<div class="rtl-block">', unsafe_allow_html=True)
        st.markdown('<b>الإرشادات بالعربية</b><br>', unsafe_allow_html=True)
        st.markdown(arabic_part, unsafe_allow_html=False)
        st.markdown('</div>', unsafe_allow_html=True)




