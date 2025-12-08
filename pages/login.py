import streamlit as st

users = {
    "admin": {"password": "123", "role": "admin"},
    "farmer": {"password": "123", "role": "farmer"},
}

def show():
    st.title("🌿 Nabta Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.role = users[username]["role"]

            if st.session_state.role == "admin":
                st.session_state.page = "admin"
            else:
                st.session_state.page = "farmer"
        else:
            st.error("Wrong username or password")

