import streamlit as st
import os

# --- Configuration for Admin Credentials ---
# It's highly recommended to use Streamlit secrets for production.
# For local development, you can set defaults or define them here.
ADMIN_USERNAME = st.secrets.get("ADMIN_USERNAME", "admin") # Default 'admin' if not found in secrets
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123") # Default 'admin123' if not found in secrets

# --- Global Streamlit Setup (only done once per app run) ---
st.set_page_config(
    page_title="SNMD Prediction App",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded"
)

# Initialize session state for login status
# Streamlit's session_state persists variables across reruns.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- Sidebar Navigation and Login Form ---
st.sidebar.title("SNMD Application")
st.sidebar.markdown("---")

if st.session_state.logged_in:
    # If logged in, show logout button and page links
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun() # Rerun the app to show the login page again

    st.sidebar.page_link("pages/1_Questionnaire_App.py", label="❓ User Questionnaire", icon="📝")
    st.sidebar.page_link("pages/2_Admin_Dashboard.py", label="📊 Admin Dashboard", icon="⚙️")
else:
    # If not logged in, show the login form
    st.sidebar.subheader("Admin Login")
    username_input = st.sidebar.text_input("Username", key="login_username")
    password_input = st.sidebar.text_input("Password", type="password", key="login_password")

    if st.sidebar.button("Login"):
        if username_input == ADMIN_USERNAME and password_input == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.username = username_input
            st.sidebar.success("Logged in successfully!")
            st.experimental_rerun() # Rerun to update the sidebar and show app pages
        else:
            st.session_state.logged_in = False # Ensure this is false on failed attempt
            st.sidebar.error("Invalid username or password")

# --- Main Page Content (Visible if not logged in or as a landing page) ---
if not st.session_state.logged_in:
    st.title("Welcome to the Social Network Mental Disorder Prediction App")
    st.info("Please login using the sidebar to access the admin dashboard or fill the questionnaire.")
    st.markdown("---")
    st.write("This application provides a tool to assess potential risks of Social Network Mental Disorders (SNMD).")
    st.write("Users can complete a psychological questionnaire, while administrators can view aggregated data and model insights from the social network analysis.")
    st.write("To proceed to the user questionnaire, simply navigate to `pages/1_Questionnaire_App.py` directly or after logging in.")