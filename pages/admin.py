import streamlit as st

def show():
    st.title("🧑‍💼 Admin Dashboard")

    st.write("Welcome admin!")

    # view all scans stored in session
    if "history" in st.session_state:
        st.subheader("All scans")
        for item in st.session_state.history:
            st.write(item)

    # fake export button
    if st.button("Export CSV"):
        st.success("Fake CSV exported 🤣")

    # fake farmer management
    st.subheader("Manage farmers")
    st.write("Fake farmer list")

    # logout
    if st.button("Logout"):
        st.session_state.page = "login"

