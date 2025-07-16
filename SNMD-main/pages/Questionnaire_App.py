import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
# CORRECTED path for questionnaire_model.pkl based on your file structure:
# From 'pages' (current location), go up one level (..) to 'SNMD-main', then into 'models', etc.
model_path = os.path.join(script_dir, "..", "models", "questionnaire_model", "questionnaire_model.pkl")


# Load trained model
model = None # Initialize model to None
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # st.success(f"Model loaded successfully from: {model_path}") # For debugging
except FileNotFoundError:
    st.error(f"Error: Questionnaire model not found at `{model_path}`. Please ensure your model training script (e.g., part of `main.py` or a dedicated `train_questionnaire_model.py` if you create one) has been run to create this file.")
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
st.subheader("📘 DSM-5 Criteria (Relates to Cyber-Relationship Addiction - CRA)")
dsm_questions = [
    "Do you feel anxious when you cannot access social media?",
    "Have you tried to cut down social media use but failed?",
    "Do you neglect responsibilities due to social media?",
    "Do you prefer virtual interactions over real ones?",
    "Do you feel restless when not using social platforms?",
]
dsm_responses = []
for i, q in enumerate(dsm_questions):
    # Changed to st.radio to make questions compulsory by forcing a "Yes" or "No" choice
    # index=None makes it initially unselected
    response = st.radio(f"{i+1}. {q}", options=["Yes", "No"], key=f"dsm_radio_{i}", index=None)
    dsm_responses.append(response)

# Calculate dsm_score based on "Yes" answers
dsm_score = sum(1 for r in dsm_responses if r == "Yes")
max_dsm = 5 # Defined max score for normalization

# --- Section 2: IAT ---
st.subheader("🌐 Internet Addiction Test (IAT) (Relates to Net Compulsion - NC)")
iat_questions = [
    "You find yourself staying online longer than intended.",
    "You feel preoccupied with social media use.",
    "Others complain about your social media habits.",
    "You check your social media before anything else.",
    "You lose sleep because of late-night Browse.",
]
# Max score for IAT is 5*5=25, min is 1*5=5
iat_responses = [st.slider(q, 1, 5, 3, key=f"iat_{i}") for i, q in enumerate(iat_questions)]
iat_score = sum(iat_responses)
max_iat = 25 # Defined max score for normalization

# --- Section 3: DASS ---
st.subheader("💬 DASS (Depression, Anxiety, Stress Scale) (Relates to Information Overload - IO)")
dass_questions = [
    "I felt down-hearted and blue (depression).",
    "I was aware of dryness of my mouth (anxiety).",
    "I found it hard to wind down. (Stress)",
    "I felt scared without any good reason. (Anxiety)",
    "I couldn’t seem to experience positive feelings. (Depression)",
]
# Max score for DASS is 3*5=15, min is 0*5=0
dass_responses = [st.slider(q, 0, 3, 1, key=f"dass_{i}") for i, q in enumerate(dass_questions)]
dass_score = sum (dass_responses)
max_dass = 15 # Defined max score for normalization


# --- Prediction Section ---
if st.button("🔍 Predict SNMD Risk"):
    if model is None:
        st.error("Prediction cannot be made: Model is not loaded.")
        st.stop()

    # Added: Validation to ensure all DSM-5 radio questions have been answered
    if any(r is None for r in dsm_responses):
        st.warning("Please answer all questions in the DSM-5 Criteria section.")
        st.stop()

    # Existing: Validation for overall minimal engagement across all sections
    # If all scores are at their absolute minimums, it suggests the user hasn't
    # provided any meaningful input beyond the default or lowest possible values.
    if dsm_score == 0 and iat_score == 5 and dass_score == 0:
        st.warning("Please fill out all sections of the questionnaire to get a meaningful risk prediction. It seems like you haven't provided any answers or all answers are at their lowest possible values across all tests.")
        st.stop() # Stop further execution if questionnaire is not sufficiently filled


    # Reshape features to (1, N_FEATURES) for model.predict
    features = np.array([[dsm_score, iat_score, dass_score]])

    try:
        # Get probabilities for each class
        # probas[0] is probability of class 0 (Low Risk)
        # probas[1] is probability of class 1 (High Risk)
        probas = model.predict_proba(features)[0]
        prob_low_risk = probas[0]
        prob_high_risk = probas[1]

        # Define the threshold for high risk (75% as requested)
        HIGH_RISK_THRESHOLD_PERCENT = 75.0

        # --- Subtype Breakdown percentages (Contribution) ---
        normalized_dsm_score = dsm_score / max_dsm if max_dsm > 0 else 0
        normalized_iat_score = iat_score / max_iat if max_iat > 0 else 0
        normalized_dass_score = dass_score / max_dass if max_dass > 0 else 0

        total_normalized_score_contribution = normalized_dsm_score + normalized_iat_score + normalized_dass_score

        if total_normalized_score_contribution == 0:
            cra_pct = 0
            nc_pct = 0
            io_pct = 0
        else:
            cra_pct = (normalized_dsm_score / total_normalized_score_contribution) * 100
            nc_pct = (normalized_iat_score / total_normalized_score_contribution) * 100
            io_pct = (normalized_dass_score / total_normalized_score_contribution) * 100

        st.markdown("## 🔎 Prediction Result:")
        st.markdown(
            f"""
        - **Cyber-Relationship Addiction (CRA) Contribution: `{cra_pct:.1f}%`**
        - **Net Compulsion (NC) Contribution: `{nc_pct:.1f}%`**
        - **Information Overload (IO) Contribution: `{io_pct:.1f}%`**
        """
        )

        # Calculate average contribution percentage
        average_contribution_pct = (cra_pct + nc_pct + io_pct) / 3 if (cra_pct + nc_pct + io_pct) > 0 else 0

        st.markdown(f"**Average Subtype Contribution: `{average_contribution_pct:.1f}%`**")


        # Pie Chart
        fig, ax = plt.subplots()
        if total_normalized_score_contribution == 0:
            ax.pie([1], labels=["No Scores"], colors=['lightgray'], autopct='%1.1f%%', startangle=90)
        else:
            pie_data = [cra_pct, nc_pct, io_pct]
            labels = ["CRA", "NC", "IO"]
            filtered_pie_data = [d for d in pie_data if d > 0]
            filtered_labels = [l for l, d in zip(labels, pie_data) if d > 0]

            if filtered_pie_data:
                ax.pie(filtered_pie_data, labels=filtered_labels, autopct="%1.1f%%", startangle=90)
            else:
                ax.pie([1], labels=["No Contributions"], colors=['lightgray'], autopct='%1.1f%%', startangle=90)

        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

        # --- Enhanced Risk Feedback based on the 75% probability threshold ---
        is_high_risk_model = (prob_high_risk * 100) >= HIGH_RISK_THRESHOLD_PERCENT
        is_high_risk_individual_contribution = (cra_pct >= HIGH_RISK_THRESHOLD_PERCENT or
                                                nc_pct >= HIGH_RISK_THRESHOLD_PERCENT or
                                                io_pct >= HIGH_RISK_THRESHOLD_PERCENT)
        is_high_risk_average_contribution = average_contribution_pct >= HIGH_RISK_THRESHOLD_PERCENT

        high_risk_reasons = []
        if is_high_risk_model:
            high_risk_reasons.append(f"Overall model prediction ({prob_high_risk * 100:.2f}%)")
        if cra_pct >= HIGH_RISK_THRESHOLD_PERCENT:
            high_risk_reasons.append(f"High Cyber-Relationship Addiction (CRA) contribution ({cra_pct:.1f}%)")
        if nc_pct >= HIGH_RISK_THRESHOLD_PERCENT:
            high_risk_reasons.append(f"High Net Compulsion (NC) contribution ({nc_pct:.1f}%)")
        if io_pct >= HIGH_RISK_THRESHOLD_PERCENT:
            high_risk_reasons.append(f"High Information Overload (IO) contribution ({io_pct:.1f}%)")
        if is_high_risk_average_contribution and not (is_high_risk_model or is_high_risk_individual_contribution):
            high_risk_reasons.append(f"High average subtype contribution ({average_contribution_pct:.1f}%)")
        
        high_risk_reasons = list(set(high_risk_reasons))
        reasons_text = " and ".join(high_risk_reasons) if high_risk_reasons else ""


        if is_high_risk_model or is_high_risk_individual_contribution or is_high_risk_average_contribution:
            st.markdown(
                f"""
            <div style='background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 6px solid #f44336;'>
                <h3 style='color: #d32f2f; margin-top: 0;'>🔴 High Risk of SNMD Detected ({prob_high_risk * 100:.2f}%) </h3>
                <p>Your assessment indicates significant risk factors for Social Network Mental Disorders. It's recommended to take this seriously.</p>
            </div>
            """, unsafe_allow_html=True,
            )

            with st.expander("🔍 Detailed Risk Factors and Recommendations", expanded=True):
                if reasons_text:
                    st.markdown(f"**Reasons for High Risk Classification:** {reasons_text}.")
                else:
                    st.markdown("**Reasons for High Risk Classification:** Based on overall assessment.")


                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                    **Your Scores (Raw):**
                    - DSM-5 Score (Relates to CRA): {dsm_score}/{max_dsm}
                    - IAT Score (Relates to NC): {iat_score}/{max_iat}
                    - DASS Score (Relates to IO): {dass_score}/{max_dass}
                    """
                    )
                with col2:
                    st.markdown(
                        f"""
                    **Contribution to Overall Score:**
                    - Cyber-Relationship Addiction: {cra_pct:.1f}%
                    - Net Compulsion: {nc_pct:.1f}%
                    - Information Overload: {io_pct:.1f}%
                    - Average Subtype Contribution: {average_contribution_pct:.1f}%
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
        else:
            st.success(
                f"""
            🟢 Low risk of SNMD ({prob_low_risk * 100:.2f}%) 
            Your assessment indicates healthy social media usage patterns. You show a low risk for Social Network Mental Disorders.
            **Average Subtype Contribution:*
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
        values = [dsm_score, iat_score, dass_score]
        
        max_values = [max_dsm, max_iat, max_dass]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(categories, values, color=["skyblue", "orange", "lightgreen"])

        ax.set_ylim(0, max(max_values) + 2)
        ax.set_ylabel("Score")
        ax.set_title("Your Scores Compared to Maximum Possible")

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                            f"{height}/{max_values[i]}", ha="center", va="bottom", fontsize=10)

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("For a professional assessment, please consult a mental health expert.")