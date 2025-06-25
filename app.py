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

    # ❗ Simulated subtype breakdown (replace with real model output when available)
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
    
    # Optional: Add a pie chart
    fig, ax = plt.subplots()
    ax.pie([cra_pct, nc_pct, io_pct], labels=["CRA", "NC", "IO"], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    if prediction == 1:
        st.error(f"🟥 High risk of SNMD ({probability * 100:.2f}%)")
    else:
        st.success(f"🟩 Low risk of SNMD ({probability * 100:.2f}%)")


    # --- Graph: Score Breakdown ---
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

# --- Divider ---
st.markdown("---")
st.header("📊 Explore SNMD Trends in Dataset")

# --- Load Dataset via File Uploader ---
st.title("📊 SNMDD Behavioral Insights Dashboard")
st.markdown("Upload your `snmdd_dataset_cleaned.csv` file below:")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # --- Preprocess ---
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date

        df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
        df['comments'] = pd.to_numeric(df['comments'], errors='coerce')
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
        df['engagement'] = df['likes'] + df['comments']

        df.dropna(subset=['engagement', 'sentiment'], inplace=True)

        # --- Heuristics for Symptoms ---
        df['negative_sentiment'] = df['sentiment'] < 0
        df['high_engagement'] = df['engagement'] > df['engagement'].mean()
        df['symptom_score'] = df[['negative_sentiment', 'high_engagement']].sum(axis=1)

        summary = df.groupby('user_id')[['negative_sentiment', 'high_engagement']].sum()
        summary['symptom_type'] = summary.apply(
            lambda row: 'Both' if row['negative_sentiment'] > 0 and row['high_engagement'] > 0
            else ('Negative Sentiment' if row['negative_sentiment'] > 0
                  else ('High Engagement' if row['high_engagement'] > 0 else 'None')),
            axis=1
        )

        # --- Quick Metrics ---
        st.subheader("📈 Project Overview")
        st.metric("Total Users", len(summary))
        st.metric("Total Posts Analyzed", len(df))

        # --- Pie Chart: Symptom Type Distribution ---
        st.subheader("🥧 Symptom Type Distribution (Pie Chart)")
        counts = summary['symptom_type'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # --- Bar Chart ---
        st.subheader("📊 User Symptom Breakdown (Bar Chart)")
        fig2, ax2 = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, palette='pastel', ax=ax2)
        ax2.set_ylabel("User Count")
        ax2.set_title("Symptom Type Distribution")
        st.pyplot(fig2)

        # --- Heatmap: Posting Frequency ---
        st.subheader("🕒 Posting Frequency Heatmap (Hour vs Date)")
        heatmap_data = df.groupby(['hour', 'date']).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax3)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Hour of Day")
        ax3.set_title("Post Frequency by Hour and Date")
        st.pyplot(fig3)

    except Exception as e:
        st.error("❌ Something went wrong while processing the file.")
        st.exception(e)

else:
    st.info("📂 Please upload your `snmdd_dataset_cleaned.csv` to begin.")