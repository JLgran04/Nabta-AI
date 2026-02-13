import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv

# Auth
from db import create_table
from auth import create_user, login_user, get_all_users, delete_user

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Nabta AI", page_icon="🌿", layout="wide")

# -------------------------
# Database
# -------------------------
create_table()

# -------------------------
# Session State Defaults
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "selected_role" not in st.session_state:
    st.session_state.selected_role = None
if "username" not in st.session_state:
    st.session_state.username = None

# -------------------------
# Custom UI Styles
# -------------------------
st.markdown("""
<style>
body { background-color: #f7f8fa; font-family: "Inter", sans-serif; }
.main-header { background: linear-gradient(90deg,#2e7d32 0%,#66bb6a 100%);
    color:white; padding:1.2rem 2rem; border-radius:12px; text-align:center; margin-bottom:1.5rem; }
.card { background:#fff; border-radius:14px; padding:1rem; border:1px solid #e5e7eb; box-shadow:0 6px 24px rgba(0,0,0,0.04);}
.result-card { background: linear-gradient(90deg,#2e7d32 0%,#66bb6a 100%); border-radius:14px; padding:1rem; color:white; margin-bottom:1rem;}
.advice-wrapper { background:#fff; border:2px solid #2e7d32; border-radius:10px; padding:1rem; margin-bottom:1rem;}
.rtl-block { direction:rtl; text-align:right; background:#fff; border-radius:10px; border:2px solid #2e7d32; padding:.9rem; margin-top:.75rem; }
.analyze-button button { width:100% !important; border-radius:10px !important; font-weight:600 !important; background: linear-gradient(90deg,#2e7d32 0%,#66bb6a 100%) !important; color:white !important; }
.warning-box { background:#fff7ed; border:1px solid #fdba74; color:#9a3412; border-radius:10px; padding:.8rem 1rem; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
<h1>🌿 Nabta AI</h1>
<p>Creating a healthier, greener, and sustainable environment in Kuwait.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Environment / Gemini AI
# -------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None

# -------------------------
# Load Models
# -------------------------
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
    0:"Corn (Gray leaf spot)",1:"Corn (Common rust)",2:"Corn (Northern Leaf Blight)",
    3:"Corn (Healthy)",4:"Pepper (Bacterial spot)",5:"Pepper (Healthy)",
    6:"Potato (Early blight)",7:"Potato (Late blight)",8:"Potato (Healthy)",
    10:"Strawberry (Leaf scorch)",11:"Strawberry (Healthy)",
    12:"Tomato (Bacterial spot)",13:"Tomato (Early blight)",14:"Tomato (Late blight)",
    15:"Tomato (Leaf Mold)",16:"Tomato (Septoria leaf spot)",17:"Tomato (Spider mites / Two-spotted spider mite)",
    18:"Tomato (Target Spot)",19:"Tomato (Yellow Leaf Curl Virus)",20:"Tomato (Mosaic virus)",21:"Tomato (Healthy)"
}

# -------------------------
# Image Preprocessing
# -------------------------
def preprocess_image(img: Image.Image, target_size=(150,150)):
    img = img.resize(target_size)
    arr = np.array(img).astype("float32")/255.0
    return np.expand_dims(arr, axis=0)

def predict_soil(img):
    if soil_model is None: return f"[Soil model not loaded: {soil_model_error}]", 0.0
    preds = soil_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    return soil_class_labels.get(idx,"Unknown"), float(preds[0][idx])

def predict_plant(img):
    if plant_model is None: return f"[Plant model not loaded: {plant_model_error}]",0.0
    preds = plant_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    return plant_class_labels.get(idx,"Unknown"), float(preds[0][idx])

def explain_prediction(label, category):
    if not gemini_model:
        return "🌐 Gemini not configured."
    prompt = (
        f"You are an experienced agricultural advisor. The AI predicted {category} = '{label}'.\n"
        "Explain what it means, actions, prevention. Split English / Arabic."
    )
    try:
        resp = gemini_model.generate_content(prompt)
        if resp.candidates and resp.candidates[0].content.parts:
            return resp.candidates[0].content.parts[0].text.strip()
        return "No explanation generated."
    except: return "Gemini explanation unavailable."

# -------------------------
# Pages
# -------------------------
def show_role_selector():
    st.markdown("### Select Role")
    role = st.radio("Continue as:", ["User", "Admin"])
    if st.button("Continue", key="role_btn"):
        st.session_state.selected_role = role
        st.experimental_rerun()

def show_auth_page():
    st.markdown(f"### {st.session_state.selected_role} Login/Register")
    menu = ["Login","Register"] if st.session_state.selected_role=="User" else ["Login"]
    choice = st.radio("Select option:", menu, key="auth_radio")

    username = st.text_input("Username", key="user_input")
    password = st.text_input("Password", type="password", key="pass_input")

    if choice=="Register" and st.button("Register", key="btn_reg"):
        if create_user(username,password,role="user"):
            st.success("Account created! Login now.")
        else: st.error("Username exists.")

    if choice=="Login" and st.button("Login", key="btn_login"):
        role = login_user(username,password)
        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.username = username
            st.experimental_rerun()
        else:
            st.error("Invalid credentials.")

def show_admin_page():
    st.sidebar.success("👑 Admin Panel")
    if st.sidebar.button("Logout"): 
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.selected_role = None
        st.session_state.username = None
        st.experimental_rerun()

    st.title("Admin Dashboard")
    st.subheader("Manage Users")
    users = get_all_users()
    for user in users:
        user_id, username, role = user
        col1,col2,col3 = st.columns([3,2,1])
        col1.write(username)
        col2.write(role)
        if role!="admin":
            if col3.button("Delete", key=f"del_{user_id}"):
                delete_user(user_id)
                st.experimental_rerun()

def show_user_page():
    st.sidebar.success("🌿 User Dashboard")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.selected_role = None
        st.session_state.username = None
        st.experimental_rerun()

    # -------------------------
    # Nabta AI Analyzer
    # -------------------------
    left_col,right_col = st.columns([1,1], gap="large")
    img = None
    with left_col:
        st.markdown('<div class="card"><h3>📥 Input Image</h3>', unsafe_allow_html=True)
        input_method = st.radio("Provide image:",["Upload","Camera"], key="input_method")
        if input_method=="Upload":
            uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
            if uploaded: img = Image.open(uploaded).convert("RGB")
        else:
            cam_img = st.camera_input("Take Photo")
            if cam_img: img = Image.open(cam_img).convert("RGB")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card"><h3>Preview & Task</h3>', unsafe_allow_html=True)
        if img: st.image(img, use_container_width=True)
        else: st.markdown('<div class="warning-box">No image yet.</div>', unsafe_allow_html=True)
        task_type = st.radio("Select Task", ["Soil Moisture","Plant Disease"], key="task_radio")
        st.markdown('</div>', unsafe_allow_html=True)

    if img and st.button("Analyze Image", key="analyze_btn"):
        with st.spinner("Analyzing..."):
            if task_type=="Soil Moisture": label, prob = predict_soil(img)
            else: label, prob = predict_plant(img)
            explanation_raw = explain_prediction(label, task_type.lower())

        english_part, arabic_part = "",""
        if "### Arabic" in explanation_raw:
            parts = explanation_raw.split("### Arabic")
            english_part = parts[0].replace("### English","").strip()
            arabic_part = parts[1].strip()
        else: english_part = explanation_raw

        st.markdown(f'<div class="result-card"><b>Prediction:</b> {label}<br>Confidence: {prob:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="advice-wrapper"><b>English Guidance</b><br>'+english_part+'</div>', unsafe_allow_html=True)
        if arabic_part: st.markdown('<div class="rtl-block"><b>الإرشادات بالعربية</b><br>'+arabic_part+'</div>', unsafe_allow_html=True)

# -------------------------
# Routing
# -------------------------
if st.session_state.selected_role is None:
    show_role_selector()
elif not st.session_state.logged_in:
    show_auth_page()
else:
    if st.session_state.role=="admin": show_admin_page()
    elif st.session_state.role=="user": show_user_page()
