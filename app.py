import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv
from db import create_table, get_all_users, delete_user
from auth import create_user, login_user  # login_user must return the user's role

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
if "username" not in st.session_state:
    st.session_state.username = ""

# -------------------------------------------------
# Auth Functions
# -------------------------------------------------
def show_auth_page():
    st.title("🌿 Nabta AI Login / Register")
    choice = st.radio("Select Option", ["Login", "Register"], key="auth_radio")

    if choice == "Register":
        username = st.text_input("Username", key="reg_user")
        password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create Account", key="reg_btn"):
            if create_user(username, password, role="user"):
                st.success("Account created successfully! You can now login.")
            else:
                st.error("Username already exists.")

    else:  # Login
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            role = login_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.username = username
            else:
                st.error("Invalid credentials")

# -------------------------------------------------
# Admin Page
# -------------------------------------------------
def show_admin_page():
    st.sidebar.success(f"👑 Admin: {st.session_state.username}")
    if st.sidebar.button("Logout", key="admin_logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = ""

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
                st.experimental_rerun()  # safe here because it's a button click

# -------------------------------------------------
# User Page
# -------------------------------------------------
def show_user_page():
    st.sidebar.success(f"🌿 User: {st.session_state.username}")
    if st.sidebar.button("Logout", key="user_logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = ""

    st.title("Nabta AI Application")
    st.write("Welcome! Upload an image to analyze soil moisture or plant disease.")

    # ---------------- Image Input ----------------
    left_col, right_col = st.columns([1,1])
    with left_col:
        input_method = st.radio("Input Method", ["Upload", "Camera"], key="input_method")
        img = None
        if input_method == "Upload":
            uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="file_uploader")
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Preview", use_container_width=True)
        else:
            cam_img = st.camera_input("Take a live photo", key="camera_input")
            if cam_img:
                img = Image.open(cam_img).convert("RGB")
                st.image(img, caption="Preview", use_container_width=True)

    with right_col:
        task_type = st.radio("Select Task", ["Soil Moisture", "Plant Disease"], horizontal=True, key="task_radio")

    # ----------------- Load Models ----------------
    soil_model = None
    plant_model = None
    soil_error = None
    plant_error = None
    try:
        soil_model = keras.models.load_model("models/soil_moisture_model.keras")
    except Exception as e:
        soil_error = str(e)
    try:
        plant_model = keras.models.load_model("models/plant_disease_model.keras")
    except Exception as e:
        plant_error = str(e)

    soil_labels = {0: "dry", 1: "moist", 2: "wet"}
    plant_labels = {
        0:"Corn (Gray leaf spot)", 1:"Corn (Common rust)", 2:"Corn (Northern Leaf Blight)", 3:"Corn (Healthy)",
        4:"Pepper (Bacterial spot)", 5:"Pepper (Healthy)", 6:"Potato (Early blight)", 7:"Potato (Late blight)",
        8:"Potato (Healthy)", 10:"Strawberry (Leaf scorch)", 11:"Strawberry (Healthy)",
        12:"Tomato (Bacterial spot)", 13:"Tomato (Early blight)", 14:"Tomato (Late blight)", 21:"Tomato (Healthy)"
    }

    # ----------------- Prediction ----------------
    def preprocess_image(img: Image.Image):
        arr = np.array(img.resize((150,150))).astype("float32")/255.0
        arr = np.expand_dims(arr,0)
        return arr

    def predict_soil(img):
        if soil_model is None:
            return f"[Soil model not loaded: {soil_error}]", 0.0
        preds = soil_model.predict(preprocess_image(img))
        idx = int(np.argmax(preds[0]))
        return soil_labels.get(idx,"Unknown"), float(preds[0][idx])

    def predict_plant(img):
        if plant_model is None:
            return f"[Plant model not loaded: {plant_error}]", 0.0
        preds = plant_model.predict(preprocess_image(img))
        idx = int(np.argmax(preds[0]))
        return plant_labels.get(idx,"Unknown"), float(preds[0][idx])

    # ----------------- Gemini ----------------
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-flash-latest")
    else:
        gemini_model = None

    def explain_prediction(label, category):
        if not gemini_model:
            return "Gemini not configured."
        prompt = (
            f"You are an agricultural advisor. Predicted {category} = {label}.\n"
            "Give explanation and advice in English and Arabic."
        )
        try:
            resp = gemini_model.generate_content(prompt)
            if resp.candidates and resp.candidates[0].content.parts:
                return resp.candidates[0].content.parts[0].text
        except:
            return "Gemini explanation unavailable."
        return "No explanation generated."

    # ----------------- Analyze Button ----------------
    if img and st.button("Analyze Image", key="analyze_btn"):
        if task_type == "Soil Moisture":
            label, prob = predict_soil(img)
            explanation = explain_prediction(label,"soil moisture")
        else:
            label, prob = predict_plant(img)
            explanation = explain_prediction(label,"plant disease")

        # Prediction card
        st.markdown(f"""
        <div style='background: linear-gradient(90deg,#2e7d32,#66bb6a); padding:1rem; border-radius:12px; color:white'>
            <h3>✅ Prediction: {label}</h3>
            <p>Confidence: {prob:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Advice
        st.text_area("Advice (English + Arabic)", explanation, height=300)
        
# ---------------------------
# Routing
# ---------------------------
if not st.session_state.logged_in:
    show_auth_page()
elif st.session_state.role == "admin":
    show_admin_page()
else:
    show_user_page()
