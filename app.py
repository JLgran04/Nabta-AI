import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv

from db import create_table
from auth import create_user, login_user, get_all_users, delete_user

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Nabta AI",
    page_icon="🌿",
    layout="wide"
)

# -----------------------
# Create Database Table
# -----------------------
create_table()

# -----------------------
# Session State
# -----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "img" not in st.session_state:
    st.session_state.img = None
if "task_type" not in st.session_state:
    st.session_state.task_type = None

# =================================================
# 🔐 AUTH / ROLE FUNCTIONS
# =================================================
def show_auth_page():
    st.title("🌿 Nabta AI Login or Register")
    menu = ["Login", "Register"]
    choice = st.radio("Select Option", menu, key="auth_choice")

    username = st.text_input("Username", key=f"{choice}_username")
    password = st.text_input("Password", type="password", key=f"{choice}_password")

    if choice == "Register" and st.button("Create Account", key="register_btn"):
        if create_user(username, password, role="user"):
            st.success("Account created successfully! You can now login.")
        else:
            st.error("Username already exists.")

    if choice == "Login" and st.button("Login", key="login_btn"):
        role = login_user(username, password)
        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.username = username
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

# -----------------------
# ADMIN PAGE
# -----------------------
def show_admin_page():
    st.sidebar.success("👑 Admin Panel")
    if st.sidebar.button("Logout", key="admin_logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.experimental_rerun()

    st.title(f"Admin Dashboard")
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

# -----------------------
# USER PAGE (Nabta AI)
# -----------------------
def show_user_page():
    st.sidebar.success(f"🌿 User Panel ({st.session_state.username})")
    if st.sidebar.button("Logout", key="user_logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.experimental_rerun()

    st.write("Welcome to Nabta AI! 🎉")

    # -----------------------
    # Header
    # -----------------------
    st.markdown(
        """
        <div style='background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 100%);
                    padding: 1.2rem;
                    border-radius: 12px;
                    text-align: center;
                    color: white;
                    margin-bottom: 1.5rem;'>
            <h1>🌿 Nabta AI</h1>
            <p>Working towards creating a healthier, greener, and sustainable environment in Kuwait.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------
    # (Insert the image upload, model prediction, and Gemini advice code here)
    # This can be the full user Nabta AI app from before
    # -----------------------
    st.info("Image analysis module goes here...")  # placeholder

# -----------------------
# ROLE SELECTION LANDING
# -----------------------
def show_role_selector():
    st.title("🌿 Nabta AI")
    st.markdown("### Select how you want to continue:")

    option = st.radio("Choose:", ["Proceed to Login/Register", "Admin"], index=0)

    if st.button("Continue", key="role_continue"):
        if option == "Admin":
            st.session_state.logged_in = True
            st.session_state.role = "admin"
            st.session_state.username = "Admin"
        else:
            show_auth_page()
        st.experimental_rerun()

# -----------------------
# ROUTING
# -----------------------
if not st.session_state.logged_in:
    show_role_selector()
else:
    if st.session_state.role == "admin":
        show_admin_page()
    elif st.session_state.role == "user":
        show_user_page()
