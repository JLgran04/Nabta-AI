import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv
from db import create_table
from auth import create_user, login_user, get_all_users, delete_user

# ----------------------------
# Page Config & DB
# ----------------------------
st.set_page_config(
    page_title="Nabta AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
create_table()

# ----------------------------
# Session State
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = ""

# ----------------------------
# GEMINI API
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None

# ----------------------------
# Load AI Models
# ----------------------------
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

soil_class_labels = {
    0: "Dry",
    1: "Moist",
    2: "Wet"
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

# ----------------------------
# Global CSS
# ----------------------------
def inject_css():
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* General */
    .stApp {
        background: #f4f8f5;
    }

    html, body, p, span, div, label, h1, h2, h3, h4, h5 {
        color: #000000 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Login/Register wrapper */
    .auth-wrap {
        max-width: 460px;
        margin: 0 auto;
    }

    .custom-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px 22px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }

    .section-title {
        font-size: 30px;
        font-weight: 700;
        color: #000000 !important;
        margin-bottom: 6px;
        text-align: left;
    }

    .center-note {
        text-align: center;
        color: #4b5563 !important;
        margin-bottom: 20px;
        font-size: 16px;
    }

    /* Make input containers shorter */
    .auth-wrap .stTextInput,
    .auth-wrap .stButton {
        max-width: 340px;
        margin-left: auto;
        margin-right: auto;
    }

    .auth-wrap .stTextInput input {
        height: 42px;
        font-size: 14px;
        border-radius: 10px !important;
    }

    /* Buttons */
    .stButton > button {
        height: 42px;
        font-size: 15px;
        border-radius: 10px;
        background: #16a34a;
        color: white !important;
        border: none;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        color: #6b7280 !important;
        font-weight: 600;
        font-size: 16px;
        padding: 6px 4px 10px 4px;
    }

    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 2px solid #16a34a !important;
    }

    /* Hide long full-width feeling */
    .auth-wrap [data-testid="stTextInputRootElement"] {
        max-width: 340px;
        margin: 0 auto;
    }

    .auth-wrap [data-testid="stButton"] {
        max-width: 340px;
        margin: 0 auto;
    }

    /* Optional: nicer placeholder */
    input::placeholder {
        color: #9ca3af !important;
        opacity: 1;
    }

    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(img: Image.Image, target_size=(150, 150)):
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_soil(img: Image.Image):
    if soil_model is None:
        return f"[Soil model not loaded: {soil_model_error}]", 0.0
    preds = soil_model.predict(preprocess_image(img), verbose=0)
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    label = soil_class_labels.get(idx, "Unknown")
    return label, prob

def predict_plant(img: Image.Image):
    if plant_model is None:
        return f"[Plant model not loaded: {plant_model_error}]", 0.0
    preds = plant_model.predict(preprocess_image(img), verbose=0)
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    label = plant_class_labels.get(idx, "Unknown")
    return label, prob

def explain_prediction(label: str, category: str) -> str:
    if not gemini_model:
        return "🌐 Gemini is not configured. Add GEMINI_API_KEY."

    prompt = (
        f"You are an experienced agricultural advisor. "
        f"The AI predicted {category} = '{label}'. "
        f"Give a clear explanation with: "
        f"1) what it means, "
        f"2) what actions the user should take, "
        f"3) prevention tips. "
        f"Format exactly with these headings:\n"
        f"### English Explanation\n"
        f"...\n"
        f"### Arabic Explanation\n"
        f"..."
    )
    try:
        resp = gemini_model.generate_content(prompt)
        if resp.candidates and resp.candidates[0].content.parts:
            text = resp.candidates[0].content.parts[0].text
            return text.strip() if text else "No explanation generated."
        return "No explanation generated."
    except Exception as e:
        return f"Gemini explanation unavailable: {e}"

def split_explanation(explanation: str):
    english_part, arabic_part = "", ""
    if "### Arabic Explanation" in explanation:
        parts = explanation.split("### Arabic Explanation")
        english_part = parts[0].replace("### English Explanation", "").strip()
        arabic_part = parts[1].strip()
    else:
        english_part = explanation.strip()
    return english_part, arabic_part

def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = ""
    st.rerun()

# ----------------------------
# UI Pages
# ----------------------------
def show_auth_page():
    st.markdown("""
        <div class="auth-wrap">
            <div class="hero">
                <h1>🌿 Nabta AI</h1>
                <p>
                    Smart agricultural assistant for soil moisture and plant disease detection.
                    Clean, fast, and designed to support better farming decisions.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Welcome</div>', unsafe_allow_html=True)
    st.markdown('<div class="center-note">Login or create a new account to continue</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_user", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")

        if st.button("Login", key="login_btn"):
            role = login_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        username = st.text_input("Create Username", key="reg_user", placeholder="Choose a username")
        password = st.text_input("Create Password", type="password", key="reg_pass", placeholder="Choose a password")

        if st.button("Create Account", key="reg_btn"):
            if username.strip() and password.strip():
                if create_user(username, password, role="user"):
                    st.success("Account created successfully. You can login now.")
                else:
                    st.error("Username already exists.")
            else:
                st.warning("Please fill in all fields.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_admin_page():
    with st.sidebar:
        st.markdown(f"### 👑 Admin Panel")
        st.markdown(f"**User:** {st.session_state.username}")
        st.markdown("---")
        if st.button("Logout", key="admin_logout"):
            logout()

    st.markdown("""
        <div class="hero">
            <h1>Admin Dashboard</h1>
            <p>Manage user accounts and keep the platform organized and secure.</p>
        </div>
    """, unsafe_allow_html=True)

    users = get_all_users()

    total_users = len(users)
    total_admins = sum(1 for _, _, role in users if role == "admin")
    total_regular = sum(1 for _, _, role in users if role == "user")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Users", total_users)
    c2.metric("Admins", total_admins)
    c3.metric("Regular Users", total_regular)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Manage Users</div>', unsafe_allow_html=True)

    header1, header2, header3 = st.columns([3, 2, 1])
    header1.markdown("**Username**")
    header2.markdown("**Role**")
    header3.markdown("**Action**")
    st.markdown("---")

    for user in users:
        user_id, username, role = user
        col1, col2, col3 = st.columns([3, 2, 1])
        col1.write(username)
        col2.write(role)

        if role != "admin":
            if col3.button("Delete", key=f"del_{user_id}"):
                delete_user(user_id)
                st.success(f"User '{username}' deleted.")
                st.rerun()
        else:
            col3.write("—")

    st.markdown('</div>', unsafe_allow_html=True)

def show_user_page():
    with st.sidebar:
        st.markdown("### 🌿 Nabta AI")
        st.markdown(f"**Welcome, {st.session_state.username}**")
        st.markdown("---")
        st.markdown("#### Features")
        st.markdown("- Soil Moisture Detection")
        st.markdown("- Plant Disease Detection")
        st.markdown("- AI Guidance in English & Arabic")
        st.markdown("---")
        if st.button("Logout", key="user_logout"):
            logout()

    st.markdown("""
        <div class="hero">
            <h1>Nabta AI Application</h1>
            <p>
                Upload or capture an image, choose the analysis type,
                and receive an AI-powered result with practical guidance.
            </p>
        </div>
    """, unsafe_allow_html=True)

    chip_col1, chip_col2, chip_col3 = st.columns([1, 1, 2])
    with chip_col1:
        st.markdown('<span class="info-chip">Modern Dashboard</span>', unsafe_allow_html=True)
    with chip_col2:
        st.markdown('<span class="info-chip">Fast Analysis</span>', unsafe_allow_html=True)
    with chip_col3:
        st.markdown('<span class="info-chip">English + Arabic Guidance</span>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    img = None

    with left_col:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📥 Input Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Choose how you want to provide the image.</div>', unsafe_allow_html=True)

        input_method = st.radio(
            "Provide image:",
            ["Upload", "Camera"],
            key="input_method",
            horizontal=True,
            label_visibility="visible"
        )

        if input_method == "Upload":
            uploaded = st.file_uploader(
                "Upload soil or plant image",
                type=["jpg", "jpeg", "png"],
                key="upload_img"
            )
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
        else:
            cam_img = st.camera_input("Take live photo", key="cam_img")
            if cam_img:
                img = Image.open(cam_img).convert("RGB")

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔎 Preview & Task</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Preview the image and choose the analysis type.</div>', unsafe_allow_html=True)

        if img:
            st.image(img, caption="Image Preview", use_container_width=True)
        else:
            st.markdown('<div class="warning-box">No image selected yet.</div>', unsafe_allow_html=True)

        task_type = st.radio(
            "What would you like to analyze?",
            ["Soil Moisture", "Plant Disease"],
            horizontal=True,
            key="task_type"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 1.2, 1])
    with analyze_col2:
        analyze_clicked = st.button("Analyze Image", key="analyze_btn")

    if analyze_clicked:
        if not img:
            st.warning("Please upload or capture an image first.")
            return

        with st.spinner("Analyzing image..."):
            if task_type == "Soil Moisture":
                label, prob = predict_soil(img)
                explanation = explain_prediction(label, "soil moisture")
            else:
                label, prob = predict_plant(img)
                explanation = explain_prediction(label, "plant disease")

        english_part, arabic_part = split_explanation(explanation)

        st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Analysis Result</div>
                <div class="result-label">✅ {label}</div>
                <div class="confidence-badge">Confidence: {prob:.2%}</div>
            </div>
        """, unsafe_allow_html=True)

        result_c1, result_c2 = st.columns(2)
        with result_c1:
            st.metric("Prediction", label)
        with result_c2:
            st.metric("Confidence Score", f"{prob:.2%}")

        st.markdown('<div class="guide-box">', unsafe_allow_html=True)
        st.markdown('<div class="guide-title">English Guidance</div>', unsafe_allow_html=True)
        st.markdown(english_part if english_part else "No English explanation available.")
        st.markdown('</div>', unsafe_allow_html=True)

        if arabic_part:
            st.markdown('<div class="rtl-box">', unsafe_allow_html=True)
            st.markdown('<div class="guide-title">الإرشادات بالعربية</div>', unsafe_allow_html=True)
            st.markdown(arabic_part)
            st.markdown('</div>', unsafe_allow_html=True)

        if soil_model_error:
            st.info(f"Soil model note: {soil_model_error}")
        if plant_model_error:
            st.info(f"Plant model note: {plant_model_error}")

# ----------------------------
# ROUTING
# ----------------------------
if not st.session_state.logged_in:
    show_auth_page()
elif st.session_state.role == "admin":
    show_admin_page()
elif st.session_state.role == "user":
    show_user_page()


