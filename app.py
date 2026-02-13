import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv
from db import create_table
from auth import create_user, login_user, get_all_users, delete_user

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Nabta AI",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------------------------
# Database
# -------------------------------------------------
create_table()

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None

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
# Load Models
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

soil_class_labels = {0: "dry", 1: "moist", 2: "wet"}

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
# Gemini Advice
# -------------------------------------------------
def explain_prediction(label: str, category: str) -> str:
    if not gemini_model:
        return "🌐 Gemini not configured. Add GEMINI_API_KEY in Streamlit Secrets."
    prompt = (
        f"You are an experienced agricultural field advisor. "
        f"The AI predicted {category} = \"{label}\".\n\n"
        f"Explain what it means, next 24h steps, prevention, isolation or watering advice.\n"
        f"Answer in TWO sections:\n### English Explanation\n- Bullet points\n### Arabic Explanation (الفهم بالعربية)\n- نقاط واضحة وعملي جداً"
    )
    try:
        resp = gemini_model.generate_content(prompt)
        if resp.candidates and resp.candidates[0].content.parts:
            text = resp.candidates[0].content.parts[0].text
            return text.strip() if text else "No explanation generated."
        return "No explanation generated."
    except Exception as e:
        return f"Gemini unavailable: {e}"

# -------------------------------------------------
# Custom Styles
# -------------------------------------------------
st.markdown("""
<style>
body {background-color:#f7f8fa; font-family:"Inter", sans-serif;}
.main-header {background:linear-gradient(90deg,#2e7d32 0%,#66bb6a 100%);color:white;padding:1.2rem 2rem;border-radius:12px;text-align:center;margin-bottom:1.5rem;box-shadow:0 12px 30px rgba(0,0,0,0.15);}
.card {background:#fff;border-radius:14px;padding:1.2rem;border:1px solid #e5e7eb;box-shadow:0 6px 24px rgba(0,0,0,0.04);}
.result-card {background:linear-gradient(90deg,#2e7d32 0%,#66bb6a 100%);border-radius:14px;padding:1.2rem 1.5rem;color:#fff;margin-bottom:1rem;}
.advice-wrapper {background:#fff;border:2px solid #2e7d32;border-radius:10px;padding:1rem;margin-bottom:1rem;line-height:1.5rem;}
.rtl-block {direction:rtl;text-align:right;background:#fff;border:2px solid #2e7d32;border-radius:10px;padding:.9rem;margin-top:.75rem;}
.analyze-button button {width:100%!important;border-radius:10px!important;background:linear-gradient(90deg,#2e7d32 0%,#66bb6a 100%)!important;border:0;color:white!important;}
.warning-box {background:#fff7ed;border:1px solid #fdba74;color:#9a3412;border-radius:10px;padding:.8rem;margin-bottom:.5rem;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("""
<div class="main-header">
<h1>🌿 Nabta AI</h1>
<p>Creating a healthier, greener, sustainable environment in Kuwait.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Authentication Pages
# -------------------------------------------------
def show_auth_page():
    st.subheader("Login or Sign Up")
    auth_choice = st.radio("Select Option", ["Login", "Register"], key="auth_choice")
    
    if auth_choice == "Register":
        username = st.text_input("Username", key="reg_username")
        password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Create Account", key="reg_btn"):
            if create_user(username, password, role="user"):
                st.success("Account created successfully! You can now login.")
            else:
                st.error("Username already exists.")
    else:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            role = login_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.rerun_flag = True  # <-- flag to trigger rerun
            else:
                st.error("Invalid credentials")

# Outside function, top-level rerun
if st.session_state.get("rerun_flag", False):
    st.session_state.rerun_flag = False
    st.experimental_rerun()


# -------------------------------------------------
# Admin Page
# -------------------------------------------------
def show_admin_page():
    st.sidebar.success(f"👑 Admin")
    if st.sidebar.button("Logout", key="logout_admin"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.experimental_rerun()
    
    st.title("Admin Dashboard")
    st.subheader("Manage Users")
    users = get_all_users()
    for user in users:
        user_id, username, role = user
        col1, col2, col3 = st.columns([3,2,1])
        col1.write(username)
        col2.write(role)
        if role != "admin":
            if col3.button("Delete", key=f"del_{user_id}"):
                delete_user(user_id)
                st.experimental_rerun()

# -------------------------------------------------
# User Page
# -------------------------------------------------
def show_user_page():
    st.sidebar.success(f"🌿 User")
    if st.sidebar.button("Logout", key="logout_user"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.experimental_rerun()

    # -------------------------------------------------
    # Layout: Input / Preview Columns
    # -------------------------------------------------
    left_col, right_col = st.columns([1,1], gap="large")
    img = None

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>📥 Input Image</h3>', unsafe_allow_html=True)
        input_method = st.radio("Choose Image Input", ["Upload Image", "Use Camera"], key="input_method")
        if input_method == "Upload Image":
            uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="file_upload")
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
        else:
            cam_img = st.camera_input("Take a live photo", key="cam_input")
            if cam_img:
                img = Image.open(cam_img).convert("RGB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Preview & Task</h3>', unsafe_allow_html=True)
        if img:
            st.image(img, caption="Preview", use_container_width=True)
        else:
            st.markdown('<div class="warning-box">No image yet. Upload or take a photo.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What to analyze?</div>', unsafe_allow_html=True)
        task_type = st.radio("", ["Soil Moisture", "Plant Disease"], horizontal=True, key="task_type")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    analyze_clicked = False
    if img:
        with st.container():
            st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
            analyze_clicked = st.button("Analyze Image with Nabta", key="analyze_btn")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">Please provide an image first.</div>', unsafe_allow_html=True)

    if analyze_clicked and img:
        with st.spinner("Analyzing..."):
            if task_type == "Soil Moisture":
                label, prob = predict_soil(img)
                explanation_raw = explain_prediction(label, "soil moisture")
            else:
                label, prob = predict_plant(img)
                explanation_raw = explain_prediction(label, "plant disease")
        
        english_part = ""
        arabic_part = ""
        if "### Arabic Explanation" in explanation_raw:
            parts = explanation_raw.split("### Arabic Explanation")
            english_part = parts[0].replace("### English Explanation", "").strip()
            arabic_part = parts[1].strip()
        else:
            english_part = explanation_raw

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">✅ Prediction: <span style="color:#fff">{label}</span></div>
            <div class="confidence">Confidence: {prob:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="advice-wrapper">', unsafe_allow_html=True)
        st.markdown('<div class="advice-header">English Guidance</div>', unsafe_allow_html=True)
        st.markdown(english_part, unsafe_allow_html=False)
        st.markdown('</div>', unsafe_allow_html=True)

        if arabic_part:
            st.markdown('<div class="rtl-block">', unsafe_allow_html=True)
            st.markdown('<b>الإرشادات بالعربية</b><br>', unsafe_allow_html=True)
            st.markdown(arabic_part, unsafe_allow_html=False)
            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# ROUTING
# -------------------------------------------------
if not st.session_state.logged_in:
    show_auth_page()
elif st.session_state.role == "admin":
    show_admin_page()
else:
    show_user_page()

