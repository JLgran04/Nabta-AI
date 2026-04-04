import os
import io
import textwrap
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import arabic_reshaper
from bidi.algorithm import get_display

from db import create_table, save_scan, get_user_history, delete_user_history
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

# Create default admin account
try:
    create_user("admin", "admin123", role="admin")
except Exception:
    pass

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
# CSS
# ----------------------------
def inject_css():
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #f7faf8 0%, #eef6f0 100%);
        color: #1f2937;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #22c55e 100%);
        border-radius: 24px;
        padding: 34px 30px;
        color: white;
        box-shadow: 0 10px 30px rgba(15, 118, 110, 0.18);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .hero p {
        margin-top: 10px;
        margin-bottom: 0;
        font-size: 1rem;
        opacity: 0.95;
        line-height: 1.7;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.75rem;
    }

    .custom-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(15, 23, 42, 0.06);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 8px 30px rgba(2, 6, 23, 0.05);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }

    .subtle-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }

    .result-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: white;
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 12px 30px rgba(17, 24, 39, 0.18);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .result-title {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-bottom: 8px;
    }

    .result-label {
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.15);
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.92rem;
    }

    .warning-box {
        border: 1px dashed #cbd5e1;
        background: #f8fafc;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
    }

    .guide-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }

    .guide-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: #0f172a;
    }

    .rtl-box {
        direction: rtl;
        text-align: right;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        padding: 0.78rem 1rem;
        font-weight: 700;
        font-size: 0.96rem;
        background: linear-gradient(135deg, #16a34a 0%, #0f766e 100%);
        color: white;
        box-shadow: 0 8px 18px rgba(34, 197, 94, 0.22);
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(34, 197, 94, 0.28);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    .stTextInput > div > div,
    .stTextArea textarea,
    .stSelectbox > div > div,
    .stNumberInput > div > div {
        border-radius: 14px !important;
    }

    .stRadio > div {
        gap: 0.75rem;
    }

    .stRadio label,
    .stRadio span,
    .stRadio p,
    div[data-baseweb="radio"] label,
    div[data-baseweb="radio"] span {
        color: #111827 !important;
        opacity: 1 !important;
        font-weight: 500;
    }

    .muted {
        color: #64748b;
        font-size: 0.95rem;
    }

    .auth-wrap {
        max-width: 520px;
        margin: 0 auto;
    }

    .center-note {
        text-align: center;
        color: #64748b;
        margin-top: 0.4rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

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
        return "Gemini is not configured. Add GEMINI_API_KEY."

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

def image_to_bytes(img: Image.Image):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def bytes_to_image(image_bytes):
    if not image_bytes:
        return None
    return Image.open(io.BytesIO(image_bytes))

def wrap_text(text, width=85):
    if not text:
        return []
    lines = []
    for paragraph in str(text).split("\n"):
        wrapped = textwrap.wrap(paragraph, width=width)
        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append("")
    return lines

def register_pdf_fonts():
    """
    Put an Arabic font file at:
    fonts/Amiri-Regular.ttf
    """
    font_path = "fonts/Amiri-Regular.ttf"
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("ArabicFont", font_path))
        except Exception:
            pass

def shape_arabic_text(text):
    if not text:
        return ""
    reshaped_text = arabic_reshaper.reshape(str(text))
    bidi_text = get_display(reshaped_text)
    return bidi_text

def wrap_arabic_text(text, width=65):
    if not text:
        return []
    shaped = shape_arabic_text(text)
    return textwrap.wrap(shaped, width=width)

def generate_history_pdf(username, history_rows):
    register_pdf_fonts()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    margin = 40
    y = page_height - margin

    pdf.setTitle(f"{username}_history")

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, y, f"Nabta AI - Scan History for {username}")
    y -= 25

    pdf.setFont("Helvetica", 10)
    pdf.drawString(margin, y, f"Total scans: {len(history_rows)}")
    y -= 25

    arabic_font_available = "ArabicFont" in pdfmetrics.getRegisteredFontNames()

    for row in history_rows:
        scan_id, scan_type, prediction, confidence, explanation_en, explanation_ar, image_data, created_at = row

        if y < 180:
            pdf.showPage()
            y = page_height - margin

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(margin, y, f"Scan #{scan_id}")
        y -= 18

        pdf.setFont("Helvetica", 10)
        pdf.drawString(margin, y, f"Type: {scan_type}")
        y -= 14
        pdf.drawString(margin, y, f"Prediction: {prediction}")
        y -= 14
        pdf.drawString(margin, y, f"Confidence: {confidence:.2%}")
        y -= 14
        pdf.drawString(margin, y, f"Date: {created_at}")
        y -= 18

        if image_data:
            try:
                pil_img = bytes_to_image(image_data)
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                img_reader = ImageReader(img_buffer)
                max_w = 180
                max_h = 140
                img_w, img_h = pil_img.size
                scale = min(max_w / img_w, max_h / img_h)
                draw_w = img_w * scale
                draw_h = img_h * scale

                pdf.drawImage(
                    img_reader,
                    margin,
                    y - draw_h,
                    width=draw_w,
                    height=draw_h,
                    preserveAspectRatio=True,
                    mask="auto"
                )
                y -= draw_h + 12
            except Exception:
                pass

        # English Explanation
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(margin, y, "English Explanation:")
        y -= 14

        pdf.setFont("Helvetica", 9)
        for line in wrap_text(explanation_en, width=90):
            if y < 60:
                pdf.showPage()
                y = page_height - margin
                pdf.setFont("Helvetica", 9)
            pdf.drawString(margin, y, line)
            y -= 11

        # Arabic Explanation
        if explanation_ar:
            y -= 8

            if y < 80:
                pdf.showPage()
                y = page_height - margin

            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(margin, y, "Arabic Explanation:")
            y -= 16

            if arabic_font_available:
                pdf.setFont("ArabicFont", 11)
                arabic_lines = wrap_arabic_text(explanation_ar, width=65)

                for line in arabic_lines:
                    if y < 60:
                        pdf.showPage()
                        y = page_height - margin
                        pdf.setFont("ArabicFont", 11)
                    pdf.drawRightString(page_width - margin, y, line)
                    y -= 14
            else:
                pdf.setFont("Helvetica", 9)
                fallback_note = "[Arabic font missing: add fonts/Amiri-Regular.ttf]"
                pdf.drawString(margin, y, fallback_note)
                y -= 14

        y -= 20
        pdf.line(margin, y, page_width - margin, y)
        y -= 20

    pdf.save()
    buffer.seek(0)
    return buffer

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
                <h1 style="text-align: center;">🌿 Nabta AI</h1>
                <p style="text-align: center;">Working Towards A Greener Kuwait.</p>
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
        st.markdown("### Admin Panel")
        st.markdown(f"**User:** {st.session_state.username}")
        st.markdown("---")
        if st.button("Logout", key="admin_logout"):
            logout()

    st.markdown("""
        <div class="hero">
            <h1>Admin Dashboard</h1>
            <p>Manage user accounts.</p>
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
        st.markdown("- Scan History with Images")
        st.markdown("- Filtered PDF Export")
        st.markdown("---")
    
    font_exists = os.path.exists("fonts/Amiri-Regular.ttf")
       
        if st.button("Logout", key="user_logout"):
            logout()

    st.markdown("""
        <div class="hero">
            <h1>Nabta AI Application</h1>
            <p>Upload or capture an image, choose the analysis type.</p>
        </div>
    """, unsafe_allow_html=True)

    if not os.path.exists("fonts/Amiri-Regular.ttf"):
        st.info("For correct Arabic in PDF, add this file: fonts/Amiri-Regular.ttf")

    img = None
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Choose how you want to provide the image.</div>', unsafe_allow_html=True)

        input_method = st.radio(
            "Provide image:",
            ["Upload", "Camera"],
            key="user_input_method",
            horizontal=True
        )

        if input_method == "Upload":
            uploaded = st.file_uploader(
                "Upload soil or plant image",
                type=["jpg", "jpeg", "png"],
                key="user_upload_img"
            )
            if uploaded is not None:
                img = Image.open(uploaded).convert("RGB")
        else:
            cam_img = st.camera_input("Take live photo", key="user_cam_img")
            if cam_img is not None:
                img = Image.open(cam_img).convert("RGB")

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Preview & Task</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Preview the image and choose the analysis type.</div>', unsafe_allow_html=True)

        if img is not None:
            st.image(img, caption="Image Preview", use_container_width=True)
        else:
            st.markdown('<div class="warning-box">No image selected yet.</div>', unsafe_allow_html=True)

        task_type = st.radio(
            "What would you like to analyze?",
            ["Soil Moisture", "Plant Disease"],
            horizontal=True,
            key="user_task_type"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 1.2, 1])
    with analyze_col2:
        analyze_clicked = st.button("Analyze Image", key="user_analyze_btn")

    if analyze_clicked:
        if img is None:
            st.warning("Please upload or take an image first.")
            return

        with st.spinner("Analyzing image..."):
            if task_type == "Soil Moisture":
                label, prob = predict_soil(img)
                explanation = explain_prediction(label, "soil moisture")
            else:
                label, prob = predict_plant(img)
                explanation = explain_prediction(label, "plant disease")

        english_part, arabic_part = split_explanation(explanation)
        image_bytes = image_to_bytes(img)

        save_scan(
            username=st.session_state.username,
            scan_type=task_type,
            prediction=label,
            confidence=prob,
            explanation_en=english_part,
            explanation_ar=arabic_part,
            image_data=image_bytes
        )

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
    # History Log Section
    # ----------------------------
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Previous Scans / History Log</div>', unsafe_allow_html=True)

    history_filter = st.selectbox(
        "Filter history by type",
        ["All", "Soil Moisture", "Plant Disease"],
        key="history_filter"
    )

    history = get_user_history(st.session_state.username, history_filter)

    if history:
        summary_df = pd.DataFrame(
            [
                {
                    "ID": row[0],
                    "Scan Type": row[1],
                    "Prediction": row[2],
                    "Confidence": f"{row[3]:.2%}",
                    "Date": row[7]
                }
                for row in history
            ]
        )

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("### Scan Details")

        for row in history:
            scan_id, scan_type, prediction, confidence, explanation_en, explanation_ar, image_data, created_at = row

            with st.expander(f"Scan #{scan_id} - {scan_type} - {prediction} ({created_at})"):
                col1, col2 = st.columns([1, 1.2])

                with col1:
                    if image_data:
                        scan_img = bytes_to_image(image_data)
                        if scan_img:
                            st.image(scan_img, caption="Saved Scan Image", use_container_width=True)
                    else:
                        st.info("No image stored for this scan.")

                with col2:
                    st.write(f"**Type:** {scan_type}")
                    st.write(f"**Prediction:** {prediction}")
                    st.write(f"**Confidence:** {confidence:.2%}")
                    st.write(f"**Date:** {created_at}")

                    st.markdown("**English Explanation**")
                    st.write(explanation_en if explanation_en else "No English explanation saved.")

                    if explanation_ar:
                        st.markdown("**Arabic Explanation**")
                        st.write(explanation_ar)

        pdf_buffer = generate_history_pdf(st.session_state.username, history)

        st.download_button(
            label="Download History as PDF",
            data=pdf_buffer,
            file_name=f"{st.session_state.username}_scan_history.pdf",
            mime="application/pdf"
        )

    else:
        st.markdown(
            '<div class="warning-box">No previous scans found yet.</div>',
            unsafe_allow_html=True
        )

    if st.button("Clear My History", key="clear_history_btn"):
        delete_user_history(st.session_state.username)
        st.success("History cleared successfully.")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# ROUTING
# ----------------------------
if not st.session_state.logged_in:
    show_auth_page()
elif st.session_state.role == "admin":
    show_admin_page()
elif st.session_state.role == "user":
    show_user_page()
else:
    st.error("Unknown role detected.")
