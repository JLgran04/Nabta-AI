import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv
from db import create_table
from auth import create_user, login_user, get_all_users, delete_user

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Nabta AI", page_icon="🌿", layout="wide")
create_table()

# -----------------------------
# Session State
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "rerun_flag" not in st.session_state:
    st.session_state.rerun_flag = False

# -----------------------------
# Authentication Page
# -----------------------------
def show_auth_page():
    st.title("🌿 Nabta AI")
    st.markdown("### Login or Create Account")
    
    auth_choice = st.radio("Select Option", ["Login", "Register"], key="auth_choice_radio")
    
    if auth_choice == "Register":
        username = st.text_input("Username", key="reg_username")
        password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Create Account", key="reg_btn"):
            if create_user(username, password, role="user"):
                st.success("Account created successfully! You can now login.")
            else:
                st.error("Username already exists.")
    
    elif auth_choice == "Login":
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            role = login_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.rerun_flag = True
            else:
                st.error("Invalid credentials")

# -----------------------------
# Admin Dashboard
# -----------------------------
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

# -----------------------------
# User Dashboard
# -----------------------------
def show_user_page():
    st.sidebar.success("🌿 User")
    if st.sidebar.button("Logout", key="user_logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.rerun_flag = True

    st.title("User Dashboard")
    st.write("Welcome to your application 🎉")

    # -----------------------------
    # Load Models and Gemini
    # -----------------------------
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-flash-latest")
    else:
        gemini_model = None

    # Load ML models
    soil_model, plant_model = None, None
    soil_error, plant_error = None, None

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
        0: "Corn (Gray leaf spot)",
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
        21: "Tomato (Healthy)"
    }

    # -----------------------------
    # Image Preprocessing
    # -----------------------------
    def preprocess_image(img: Image.Image):
        img = img.resize((150,150))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict_soil(img):
        if soil_model is None:
            return f"[Soil model not loaded: {soil_error}]", 0.0
        preds = soil_model.predict(preprocess_image(img))
        idx = int(np.argmax(preds[0]))
        return soil_labels.get(idx, "Unknown"), float(preds[0][idx])

    def predict_plant(img):
        if plant_model is None:
            return f"[Plant model not loaded: {plant_error}]", 0.0
        preds = plant_model.predict(preprocess_image(img))
        idx = int(np.argmax(preds[0]))
        return plant_labels.get(idx, "Unknown"), float(preds[0][idx])

    # -----------------------------
    # Gemini Advice
    # -----------------------------
    def explain_prediction(label: str, category: str) -> str:
        if not gemini_model:
            return "🌐 Gemini not configured. Add GEMINI_API_KEY in Streamlit secrets."
        prompt = (
            f"You are an experienced agricultural field advisor. "
            f"The AI predicted {category} = '{label}'. Explain what it means, next steps, prevention tips.\n\n"
            f"### English Explanation\n- Use simple English.\n\n"
            f"### Arabic Explanation\n- اكتب شرحاً باللغة العربية الفصحى.\n"
        )
        try:
            resp = gemini_model.generate_content(prompt)
            if resp.candidates and resp.candidates[0].content.parts:
                text = resp.candidates[0].content.parts[0].text
                return text.strip() if text else "No explanation generated."
            return "No explanation generated."
        except Exception as e:
            return f"Gemini unavailable: {e}"

    # -----------------------------
    # Layout: Input / Preview
    # -----------------------------
    left_col, right_col = st.columns([1,1], gap="large")
    with left_col:
        st.markdown('<div class="card"><h3>📥 Input Image</h3>', unsafe_allow_html=True)
        input_method = st.radio("Choose input", ["Upload", "Camera"], key="input_method")
        img = None
        if input_method == "Upload":
            uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="upload_input")
            if uploaded: img = Image.open(uploaded).convert("RGB")
        else:
            cam_img = st.camera_input("Take a photo", key="camera_input")
            if cam_img: img = Image.open(cam_img).convert("RGB")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card"><h3>Preview & Task</h3>', unsafe_allow_html=True)
        if img: st.image(img, caption="Preview", use_container_width=True)
        else: st.markdown('<div class="warning-box">No image yet.</div>', unsafe_allow_html=True)
        task_type = st.radio("Task", ["Soil Moisture","Plant Disease"], horizontal=True, key="task_radio")
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # Analyze Button
    # -----------------------------
    analyze_clicked = False
    if img:
        analyze_clicked = st.button("Analyze Image with Nabta", key="analyze_btn")
    else:
        st.markdown('<div class="warning-box">Provide an image first.</div>', unsafe_allow_html=True)

    # -----------------------------
    # Display Results
    # -----------------------------
    if analyze_clicked and img:
        with st.spinner("Analyzing..."):
            if task_type == "Soil Moisture": label, prob = predict_soil(img)
            else: label, prob = predict_plant(img)
            explanation = explain_prediction(label, task_type)

        # Split English / Arabic
        if "### Arabic Explanation" in explanation:
            parts = explanation.split("### Arabic Explanation")
            english_part = parts[0].replace("### English Explanation","").strip()
            arabic_part = parts[1].strip()
        else:
            english_part = explanation
            arabic_part = ""

        # Prediction Card
        st.markdown(f"""
            <div class="result-card">
                <div class="result-label">✅ Prediction: <span style="color:#fff;">{label}</span></div>
                <div class="confidence">Confidence: {prob:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

        # English Advice
        st.markdown('<div class="advice-wrapper"><div class="advice-header">English Guidance</div>', unsafe_allow_html=True)
        st.markdown(english_part, unsafe_allow_html=False)
        st.markdown('</div>', unsafe_allow_html=True)

        # Arabic Advice
        if arabic_part:
            st.markdown('<div class="rtl-block"><b>الإرشادات بالعربية</b><br>', unsafe_allow_html=True)
            st.markdown(arabic_part, unsafe_allow_html=False)
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# ROUTING
# -----------------------------
if not st.session_state.logged_in:
    show_auth_page()
else:
    if st.session_state.role == "admin": show_admin_page()
    else: show_user_page()

# -----------------------------
# Safe Rerun
# -----------------------------
if st.session_state.rerun_flag:
    st.session_state.rerun_flag = False
    st.experimental_rerun()
