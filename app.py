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
st.set_page_config(page_title="Nabta AI", page_icon="🌿", layout="wide")

# -------------------------------------------------
# Create Database Table
# -------------------------------------------------
create_table()

# -------------------------------------------------
# Session State Defaults
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "rerun_flag" not in st.session_state:
    st.session_state.rerun_flag = False

# -------------------------------------------------
# Load Environment & Gemini API
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
# Gemini AI Prediction Explanation
# -------------------------------------------------
def explain_prediction(label: str, category: str) -> str:
    if not gemini_model:
        return "🌐 Gemini is not configured. Add your GEMINI_API_KEY in Streamlit Secrets."
    prompt = (
        f"You are an experienced agricultural field advisor. "
        f"The AI predicted {category} = \"{label}\".\n"
        f"Explain the meaning, actions, prevention, and watering/disease advice.\n"
        f"Answer in English and Arabic.\n"
        f"### English Explanation\n"
        f"- Use simple bullet points.\n"
        f"### Arabic Explanation (الفهم بالعربية)\n"
        f"- اكتب شرحاً تفصيلياً باللغة العربية الفصحى السهلة.\n"
    )
    try:
        resp = gemini_model.generate_content(prompt)
        if resp.candidates and resp.candidates[0].content.parts:
            text = resp.candidates[0].content.parts[0].text
            return text.strip() if text else "No explanation generated."
        return "No explanation generated (empty Gemini response)."
    except Exception as e:
        return f"Gemini explanation unavailable: {e}"

# -------------------------------------------------
# ---------------- AUTH PAGE ----------------------
# -------------------------------------------------
def show_auth_page():
    st.title("🌿 Nabta AI")
    st.markdown("### Login or Create Account")

    menu = ["Login", "Register"]
    choice = st.radio("Select Option", menu, key="auth_radio")

    if choice == "Register":
        username = st.text_input("Username", key="reg_user")
        password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create Account", key="btn_register"):
            if create_user(username, password, role="user"):
                st.success("Account created successfully! You can now login.")
            else:
                st.error("Username already exists.")

    if choice == "Login":
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="btn_login"):
            role = login_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.rerun_flag = True
            else:
                st.error("Invalid credentials")

# -------------------------------------------------
# ---------------- ADMIN PAGE ---------------------
# -------------------------------------------------
def show_admin_page():
    st.sidebar.success("👑 Admin")
    if st.sidebar.button("Logout", key="admin_logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.rerun_flag = True

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
                st.session_state.rerun_flag = True

# -------------------------------------------------
# ---------------- USER PAGE ----------------------
# -------------------------------------------------
def show_user_page():
    st.sidebar.success("🌿 User")
    if st.sidebar.button("Logout", key="user_logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.rerun_flag = True

    st.title("User Dashboard")
    st.write("Welcome to your application 🎉")

    # ----------------- Nabta AI analysis -----------------
    left_col, right_col = st.columns([1,1], gap="large")
    img = None

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>📥 Input Image</h3>', unsafe_allow_html=True)
        input_method = st.radio("Choose input method:", ["Upload Image", "Use Camera"], key="input_method")

        if input_method == "Upload Image":
            uploaded = st.file_uploader("Upload a soil or plant image", type=["jpg","jpeg","png"], key="file_upload")
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
        else:
            cam_img = st.camera_input("Take a live photo", key="camera_input")
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

        st.markdown('<div class="section-title">What do you want to analyze?</div>', unsafe_allow_html=True)
        task_type = st.radio("", ["Soil Moisture", "Plant Disease"], horizontal=True, key="task_radio")
        st.markdown('</div>', unsafe_allow_html=True)

    analyze_clicked = st.button("Analyze Image with Nabta", key="analyze_btn") if img else False

    if analyze_clicked and img:
        with st.spinner("Analyzing image and generating advice..."):
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
            <div class="result-label">
                ✅ Prediction: <span style="color:#ffffff;">{label}</span>
            </div>
            <div class="confidence">
                Confidence: {prob:.2f}
            </div>
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
# ---------------- ROUTING ------------------------
# -------------------------------------------------
if not st.session_state.logged_in:
    show_auth_page()
else:
    if st.session_state.role == "admin":
        show_admin_page()
    elif st.session_state.role == "user":
        show_user_page()

# -------------------------------------------------
# ---------------- SAFE RERUN --------------------
# -------------------------------------------------
if st.session_state.get("rerun_flag", False):
    st.session_state.rerun_flag = False
    st.experimental_rerun()
