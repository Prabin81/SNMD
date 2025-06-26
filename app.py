import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# --- Streamlit Page Setup ---
st.set_page_config(page_title="SNMD Prediction App", layout="centered")
st.title("🧠 Social Network Mental Disorder (SNMD) Prediction")
st.markdown("This app evaluates your SNMD risk based on psychological criteria from **DSM-5**, **IAT**, and **DASS** tests.")

# --- Section 1: DSM-5 ---
st.subheader("📘 DSM-5 Criteria")
dsm_questions = [
    "Do you feel anxious when you cannot access social media?",
    "Have you tried to cut down social media use but failed?",
    "Do you neglect responsibilities due to social media?",
    "Do you prefer virtual interactions over real ones?",
    "Do you feel restless when not using social platforms?"
]
dsm_responses = [st.checkbox(q) for q in dsm_questions]
dsm_score = sum(1 for r in dsm_responses if r)

# --- Section 2: IAT ---
st.subheader("🌐 Internet Addiction Test (IAT)")
iat_questions = [
    "You find yourself staying online longer than intended.",
    "You feel preoccupied with social media use.",
    "Others complain about your social media habits.",
    "You check your social media before anything else.",
    "You lose sleep because of late-night browsing."
]
iat_responses = [st.slider(q, 1, 5, 3) for q in iat_questions]
iat_score = sum(iat_responses)

# --- Section 3: DASS ---
st.subheader("💬 DASS (Depression, Anxiety, Stress Scale)")
dass_questions = [
    "I felt down-hearted and blue. (Depression)",
    "I was aware of dryness of my mouth. (Anxiety)",
    "I found it hard to wind down. (Stress)",
    "I felt scared without any good reason. (Anxiety)",
    "I couldn’t seem to experience positive feelings. (Depression)"
]
dass_responses = [st.slider(q, 0, 3, 1) for q in dass_questions]
dass_score = sum(dass_responses)

# --- Prediction Section ---
if st.button("🔍 Predict SNMD Risk"):
    features = np.array([[dsm_score, iat_score, dass_score]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Overall SNMD probability

    # Subtype Breakdown
    cra_score = min(dsm_score / 5, 1.0) * 100
    nc_score = min(iat_score / 25, 1.0) * 100
    io_score = min(dass_score / 15, 1.0) * 100

    total = cra_score + nc_score + io_score
    cra_pct = (cra_score / total) * 100
    nc_pct = (nc_score / total) * 100
    io_pct = (io_score / total) * 100

    st.markdown("## 🔎 Prediction Result:")
    st.markdown(f"""
    - **Cyber-Relationship Addiction (CRA): `{cra_pct:.1f}%`**
    - **Net Compulsion (NC): `{nc_pct:.1f}%`**
    - **Information Overload (IO): `{io_pct:.1f}%`**
    """)

    # Pie Chart
    fig, ax = plt.subplots()
    ax.pie([cra_pct, nc_pct, io_pct], labels=["CRA", "NC", "IO"], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # --- Enhanced Risk Feedback ---
    if prediction == 1:
        st.markdown(f"""
        <div style='background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 6px solid #f44336;'>
            <h3 style='color: #d32f2f; margin-top: 0;'>🔵 High Risk of SNMD Detected ({probability * 100:.2f}%)</h3>
            <p style='margin-bottom: 0;'>Your assessment indicates significant risk factors for Social Network Mental Disorders.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 Detailed Risk Factors", expanded=True):
            st.markdown("### Primary Risk Factors Identified:")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Psychological Indicators:**
                - DSM-5 Score: {dsm_score}/5  
                - IAT Score: {iat_score}/25  
                - DASS Score: {dass_score}/15
                """)
            with col2:
                st.markdown(f"""
                **Behavioral Patterns:**
                - Cyber-Relationship Addiction: {cra_pct:.1f}%  
                - Net Compulsion: {nc_pct:.1f}%  
                - Information Overload: {io_pct:.1f}%
                """)

            st.markdown("""
            ### Recommended Actions:
            - 🩺 Consult a mental health professional
            - ⏱️ Set strict screen time limits
            - 📵 Plan weekly digital detox days
            - 🧘 Practice mindfulness
            - 👥 Increase real-life social interactions
            """)

        st.warning("""
        **Important:** This tool is not a medical diagnosis. Please consult a professional for clinical assessment.
        """)

    else:
        st.success(f"""
        🟢 Low risk of SNMD ({probability * 100:.2f}%)

        Your assessment shows healthy social media usage patterns. 
        Maintain your current balanced approach and remain mindful of any changes.
        """)

    # --- Score Bar Chart ---
    st.subheader("📊 Score Breakdown")
    categories = ['DSM-5', 'IAT', 'DASS']
    values = [dsm_score, iat_score, dass_score]

    fig, ax = plt.subplots()
    bars = ax.bar(categories, values, color=['skyblue', 'orange', 'lightgreen'])
    ax.set_ylim(0, max(15, max(values) + 2))
    ax.set_ylabel("Score")
    ax.set_title("Your Questionnaire Scores")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height}', ha='center', va='bottom')
    st.pyplot(fig)

    st.markdown("**Note:** This is a supportive screening tool, not a medical diagnosis.")
st.markdown("For a professional assessment, please consult a mental health expert.")        