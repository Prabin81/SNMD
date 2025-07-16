import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "questionnaire_model", "questionnaire_model.pkl") # CORRECTED path


# Load trained model
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # st.success(f"Model loaded successfully from: {model_path}") # For debugging
except FileNotFoundError:
    st.error(f"Error: Questionnaire model not found at {model_path}. Please ensure 'train_questionnaire_model.py' has been run.")
    st.stop() # Stop the app if model is not found
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- Streamlit Page Setup ---
st.title("🧠 Social Network Mental Disorder (SNMD) Prediction")
st.markdown(
    "This app evaluates your SNMD risk based on psychological criteria from **DSM-5**, **IAT**, and **DASS** tests."
)

# --- Section 1: DSM-5 ---
st.subheader("📘 DSM-5 Criteria")
dsm_questions = [
    "Do you feel anxious when you cannot access social media?",
    "Have you tried to cut down social media use but failed?",
    "Do you neglect responsibilities due to social media?",
    "Do you prefer virtual interactions over real ones?",
    "Do you feel restless when not using social platforms?",
]
dsm_responses = [st.checkbox(q, key=f"dsm_{i}") for i, q in enumerate(dsm_questions)] # Added unique keys
dsm_score = sum (1 for r in dsm_responses if r)

# --- Section 2: IAT ---
st.subheader("🌐 Internet Addiction Test (IAT)")
iat_questions = [
    "You find yourself staying online longer than intended.",
    "You feel preoccupied with social media use.",
    "Others complain about your social media habits.",
    "You check your social media before anything else.",
    "You lose sleep because of late-night Browse.",
]
# Max score for IAT is 5*5=25, min is 1*5=5
iat_responses = [st.slider(q, 1, 5, 3, key=f"iat_{i}") for i, q in enumerate(iat_questions)] # Added unique keys
iat_score = sum(iat_responses)

# --- Section 3: DASS ---
st.subheader("💬 DASS (Depression, Anxiety, Stress Scale)")
dass_questions = [
    "I felt down-hearted and blue (depression).",
    "I was aware of dryness of my mouth (anxiety).",
    "I found it hard to wind down. (Stress)",
    "I felt scared without any good reason. (Anxiety)",
    "I couldn’t seem to experience positive feelings. (Depression)",
]
# Max score for DASS is 3*5=15, min is 0*5=0
dass_responses = [st.slider(q, 0, 3, 1, key=f"dass_{i}") for i, q in enumerate(dass_questions)] # Added unique keys
dass_score = sum (dass_responses)

# --- Prediction Section ---
if st.button("🔍 Predict SNMD Risk"):
    # Reshape features to (1, N_FEATURES) for model.predict
    features = np.array([[dsm_score, iat_score, dass_score]])

    prediction = model.predict(features)[0] # 0 or 1
    probas = model.predict_proba(features)[0] # Probabilities for each class

    # Determine the probability of the predicted class
    # Assuming class 1 is "high risk" and class 0 is "low risk"
    if prediction == 1:
        probability = probas[1]
    else:
        probability = probas[0]

    # --- Subtype Breakdown percentages ---
    # Calculate percentages as a proportion of the sum of all user-entered scores
    # This gives a sense of which category contributed most to the overall score.
    cra_score_contribution = dsm_score
    nc_score_contribution = iat_score
    io_score_contribution = dass_score

    total_score_contribution = cra_score_contribution + nc_score_contribution + io_score_contribution

    if total_score_contribution == 0: # Avoid division by zero if all scores are 0
        cra_pct = 0
        nc_pct = 0
        io_pct = 0
    else:
        cra_pct = (cra_score_contribution / total_score_contribution) * 100
        nc_pct = (nc_score_contribution / total_score_contribution) * 100
        io_pct = (io_score_contribution / total_score_contribution) * 100

    st.markdown("## 🔎 Prediction Result:")
    st.markdown(
        f"""
    - **Cyber-Relationship Addiction (CRA) Contribution: `{cra_pct:.1f}%`**
    - **Net Compulsion (NC) Contribution: `{nc_pct:.1f}%`**
    - **Information Overload (IO) Contribution: `{io_pct:.1f}%`**
    """
    )

    # Pie Chart
    fig, ax = plt.subplots()
    if total_score_contribution == 0:
        ax.pie([1], labels=["No Scores"], colors=['lightgray'], autopct='%1.1f%%', startangle=90)
    else:
        ax.pie([cra_pct, nc_pct, io_pct], labels=["CRA", "NC", "IO"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal") # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory

    # --- Enhanced Risk Feedback ---
    if prediction == 1: # Assuming 1 means high risk
        st.markdown(
            f"""
        <div style='background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 6px solid #f44336;'>
            <h3 style='color: #d32f2f; margin-top: 0;'>🔴 High Risk of SNMD Detected ({probability * 100:.2f}%) </h3>
            <p>Your assessment indicates significant risk factors for Social Network Mental Disorders. It's recommended to take this seriously.</p>
        </div>
        """, unsafe_allow_html=True,
        )

        with st.expander("🔍 Detailed Risk Factors and Recommendations", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                **Your Scores (Raw):**
                - DSM-5 Score: {dsm_score}/5
                - IAT Score: {iat_score}/25
                - DASS Score: {dass_score}/15
                """
                )
            with col2:
                st.markdown(
                    f"""
                **Contribution to Overall Score:**
                - Cyber-Relationship Addiction: {cra_pct:.1f}%
                - Net Compulsion: {nc_pct:.1f}%
                - Information Overload: {io_pct:.1f}%
                """
                )

            st.markdown(
                """
            ### Recommended Actions:
            - 🩺 **Consult a mental health professional:** For a comprehensive diagnosis and personalized support.
            - ⏱️ **Set strict screen time limits:** Use apps or phone settings to manage daily usage.
            - 📵 **Plan weekly digital detox days:** Designate periods to completely disconnect from social media.
            - 🧘 **Practice mindfulness and relaxation techniques:** To manage anxiety and stress.
            - 👥 **Increase real-life social interactions:** Prioritize face-to-face connections over virtual ones.
            - 🛌 **Improve sleep hygiene:** Ensure adequate and quality sleep, avoiding social media before bed.
            """
            )

        st.warning(
            """
        **Important Disclaimer:** This tool is a supportive screening tool and *not* a substitute for professional medical advice or diagnosis. Please consult a qualified mental health professional for clinical assessment and guidance.
        """
        )
    else: # Prediction == 0 (Low Risk)
        st.success(
            f"""
        🟢 Low risk of SNMD ({probability * 100:.2f}%)

        Your assessment indicates healthy social media usage patterns. You show a low risk for Social Network Mental Disorders.
        """
        )
        st.info(
            """
        **Recommendation:** Maintain your current balanced approach to social media. Continue to be mindful of your usage and any potential changes in your well-being. Regular self-assessment can help in maintaining mental health.
        """
        )

    # --- Score Bar Chart ---
    st.subheader("📊 Your Questionnaire Scores Breakdown")
    categories = ["DSM-5", "IAT", "DASS"]
    values = [dsm_score, iat_score, dass_score] # Raw scores for the bar chart

    # Max possible scores for each category
    max_dsm = 5
    max_iat = 25
    max_dass = 15
    max_values = [max_dsm, max_iat, max_dass]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=["skyblue", "orange", "lightgreen"])

    # Set y-axis limits to clearly show score relative to max
    ax.set_ylim(0, max(max_values) + 2) # Set y-limit based on the highest max score + buffer
    ax.set_ylabel("Score")
    ax.set_title("Your Scores Compared to Maximum Possible")

    # Add text labels for actual scores and max scores
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f"{height}/{max_values[i]}", ha="center", va="bottom", fontsize=10)

    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory

st.markdown("---") # Separator at the bottom
st.markdown("For a professional assessment, please consult a mental health expert.")