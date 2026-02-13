import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv
from db import create_table
from auth import create_user, login_user, get_all_users, delete_user

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Nabta AI",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# Initialize DB & Session State
# -------------------------------
create_table()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# -------------------------------
# Load API Keys
# -------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None

# -------------------------------
# Load Models
# -------------------------------
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

# -------------------------------
# Helper Functions
# -------------------------------
def preprocess_image(img: Image.Image, target_size=(150,150)):
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_soil(img: Image.Image):
    if soil_model is None:
        return f"[Soil model not loaded: {soil_model_error}]", 0.0
    preds = soil_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    return soil_class_labels.get(idx, "Unknown"), prob

def predict_plant(img: Image.Image):
    if plant_model is None:
        return f"[Plant model not loaded: {plant_model_error}]", 0.0
    preds = plant_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    return plant_class_labels.get(idx, "Unknown"), prob

def explain_prediction(label: str, category: str) -> str:
    if not gemini_model:
        return "🌐 Gemini not configured."
    prompt = (
        f"You are an experienced agricultural advisor. Prediction: {category} = {label}.\n"
        "Explain in English and Arabic with actionable steps."
    )
    try:
        resp = gemini_model.generate_content(prompt)
        if resp.candidates and resp.candidates[0].content.parts:
            return resp.candidates[0].content.parts[0].text.strip()
        return "No explanation generated."
    except Exception as e:
        return f"Gemini unavailable: {e}"

# -------------------------------
# AUTH PAGE
# -------------------------------
def show_auth_page():
    st.title("🌿 Nabta AI")
    st.subheader("Login or Register")
    choice = st.radio("Select Option", ["Login", "Register"], key="auth_choice")

    if choice == "Register":
        username = st.text_input("Username", key="reg_username")
        password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Create Account", key="reg_btn"):
            if create_user(username, password, role="user"):
                st.success("Account created successfully! Login now.")
            else:
                st.error("Username already exists.")
    else:  # Login
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            role = login_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.show_dashboard = True
            else:
                st.error("Invalid credentials")

# -------------------------------
# ADMIN PAGE
# -------------------------------
def show_admin_page():
    st.sidebar.success("👑 Admin")
    if st.sidebar.button("Logout", key="logout_admin"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.show_dashboard = False

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

# -------------------------------
# USER PAGE
# -------------------------------
def show_user_page():
    st.sidebar.success("🌿 User")
    if st.sidebar.button("Logout", key="logout_user"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.show_dashboard = False

    st.title("User Dashboard")
    st.write("Welcome to your application 🎉")

    # --- Image Upload and Analysis ---
    left_col, right_col = st.columns(2)
    img = None
    with left_col:
        method = st.radio("Image Input Method", ["Upload", "Camera"], key="img_method")
        if method == "Upload":
            uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"], key="upload_img")
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Preview")
        else:
            cam_img = st.camera_input("Take Photo", key="cam_img")
            if cam_img:
                img = Image.open(cam_img).convert("RGB")
                st.image(img, caption="Preview")

    with right_col:
        task_type = st.radio("Select Task", ["Soil Moisture", "Plant Disease"], key="task_radio")

    analyze_clicked = st.button("Analyze", key="analyze_btn")
    if analyze_clicked and img:
        if task_type == "Soil Moisture":
            label, prob = predict_soil(img)
            explanation = explain_prediction(label, "soil moisture")
        else:
            label, prob = predict_plant(img)
            explanation = explain_prediction(label, "plant disease")

        # Split English/Arabic
        english_text, arabic_text = "", ""
        if "### Arabic Explanation" in explanation:
            parts = explanation.split("### Arabic Explanation")
            english_text = parts[0].replace("### English Explanation","").strip()
            arabic_text = parts[1].strip()
        else:
            english_text = explanation

        st.markdown(f"**Prediction:** {label} (Confidence: {prob:.2f})")
        st.markdown("**English Guidance:**")
        st.markdown(english_text)
        if arabic_text:
            st.markdown("**الإرشادات بالعربية:**")
            st.markdown(arabic_text)

# -------------------------------
# ROUTING
# -------------------------------
if not st.session_state.logged_in or not st.session_state.show_dashboard:
    show_auth_page()
else:
    if st.session_state.role == "admin":
        show_admin_page()
    else:
        show_user_page()
