import streamlit as st

def show():
    st.title("👨‍🌾 Farmer Dashboard")

    st.write("Welcome farmer!")

    # initialize history store
    if "history" not in st.session_state:
        st.session_state.history = []

    # fake scanning button
    if st.button("Scan Plant / Soil"):
        st.session_state.history.append(
            {"type": "scan", "result": "Fake result", "note": "Sample"}
        )
        st.success("Scan stored in history!")

    # show history
    st.subheader("History")
    for item in st.session_state.history:
        st.write(item)

    # logout
    if st.button("Logout"):
        st.session_state.page = "login"
