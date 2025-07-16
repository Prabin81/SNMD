import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image # For loading images like confusion matrix
import numpy as np # For numerical operations, e.g., in describe() and correlation
import pickle # To potentially load model if needed for live admin functions

# --- Access Control ---
# This ensures that this page can only be accessed if the user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("🔒 Please log in as an admin from the home page to access this dashboard.")
    st.stop() # Stop execution of this page if not logged in

# --- Dynamic Path Handling ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# CORRECTED base_root_dir: Go up two levels from 'pages'
#   1. From 'pages' to 'SNMD-main' (..)
#   2. From 'SNMD-main' to 'SNMD-MAIN' (..)
base_root_dir = os.path.join(script_dir, "..", "..") # This now points to E:/SNMD-MAIN/

# Define absolute paths to the relevant data and model output files
LABELED_DATA_PATH = os.path.join(base_root_dir, "outputs", "user_features_labeled.csv")
CLEANED_DATA_PATH = os.path.join(base_root_dir, "outputs", "snmdd_dataset_cleaned.csv")

# ssl_results folder is directly under E:/SNMD-MAIN/outputs/
MODEL_RESULTS_DIR = os.path.join(base_root_dir, "outputs", "ssl_results")

# Paths for specific model output files within MODEL_RESULTS_DIR
CLASSIFICATION_REPORT_PATH = os.path.join(MODEL_RESULTS_DIR, "rf_classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_RESULTS_DIR, "confusion_matrix_rf.png")
FEATURE_IMPORTANCES_PATH = os.path.join(MODEL_RESULTS_DIR, "feature_importances_rf.png")
MODEL_RF_PATH = os.path.join(base_root_dir, "outputs", "model_rf.pkl") # Assuming main model is directly in E:/SNMD-MAIN/outputs/

# --- Streamlit Page Content ---
st.title("📊 Admin Dashboard: Social Network Data Insights")
st.markdown("This dashboard provides an overview of the processed social network data and machine learning model performance.")

# --- Tabbed Navigation ---
tab1, tab2, tab3 = st.tabs(["Processed Data", "Model Performance", "Raw Cleaned Data"])

# --- Tab 1: Processed Data (user_features_labeled.csv) ---
with tab1:
    st.header("🔬 Processed User Features and Labels")
    st.write("This table shows the engineered features and generated risk labels for each user, used for training the main SNMD prediction model.")

    @st.cache_data # Cache data loading for performance
    def load_labeled_data(path):
        return pd.read_csv(path)

    try:
        labeled_df = load_labeled_data(LABELED_DATA_PATH)
        st.dataframe(labeled_df)

        st.subheader("Summary Statistics of Labeled Features")
        st.write(labeled_df.describe())

        st.subheader("Distribution of Generated Risk Labels")
        fig, ax = plt.subplots()
        # Get value counts of the 'label' column
        label_counts = labeled_df['label'].value_counts().sort_index()

        # Define mapping for labels to descriptive names and corresponding colors
        label_map = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'} # Adjust if your labels differ (e.g., just 0 and 1)
        colors_map = {0: 'lightgreen', 1: 'skyblue', 2: 'lightcoral'}

        # Prepare data for plotting, ensuring all defined labels are considered
        plot_labels = [label_map.get(lbl, f'Unknown ({lbl})') for lbl in sorted(label_map.keys())]
        plot_counts = [label_counts.get(lbl_val, 0) for lbl_val in sorted(label_map.keys())]
        plot_colors = [colors_map.get(lbl, 'gray') for lbl in sorted(label_map.keys())]

        # Filter out labels that truly have zero counts if you prefer not to show them
        active_labels = [plot_labels[i] for i, count in enumerate(plot_counts) if count > 0]
        active_counts = [count for count in plot_counts if count > 0]
        active_colors = [plot_colors[i] for i, count in enumerate(plot_counts) if count > 0]


        if not active_counts: # Handle case where there's no data
            st.info("No labeled data to display distribution.")
        else:
            ax.bar(active_labels, active_counts, color=active_colors)
            ax.set_title('Distribution of Generated Risk Labels Across Users')
            ax.set_xlabel('Risk Level')
            ax.set_ylabel('Number of Users')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Correlation Matrix of Features")
        # Select only numeric columns for correlation calculation, excluding the 'label' itself
        numeric_cols = labeled_df.select_dtypes(include=np.number).columns.drop('label', errors='ignore')
        if not numeric_cols.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(labeled_df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Matrix of Engineered Features')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No numeric features found for correlation analysis in the labeled data.")


    except FileNotFoundError:
        st.error(f"Labeled data file not found at `{LABELED_DATA_PATH}`. Please ensure `label_generation.py` has been run and the file exists.")
    except pd.errors.EmptyDataError:
        st.error(f"The file `{LABELED_DATA_PATH}` is empty. Please check the data source.")
    except Exception as e:
        st.error(f"Error loading or processing labeled data: {e}")

# --- Tab 2: Model Performance ---
with tab2:
    st.header("📈 Machine Learning Model Performance (RandomForest Classifier)")
    st.write("Evaluation metrics and visualizations for the RandomForest Classifier trained on the social network data features.")

    # Load and display Classification Report
    st.subheader("Classification Report")
    try:
        with open(CLASSIFICATION_REPORT_PATH, "r") as f:
            report_content = f.read()
        st.text(report_content)
    except FileNotFoundError:
        st.error(f"Classification report file not found at `{CLASSIFICATION_REPORT_PATH}`. Please run `model_training.py` to generate model results.")
    except Exception as e:
        st.error(f"Error loading classification report: {e}")

    # Display Confusion Matrix Image
    st.subheader("Confusion Matrix")
    try:
        conf_matrix_img = Image.open(CONFUSION_MATRIX_PATH)
        st.image(conf_matrix_img, caption="Confusion Matrix for RandomForest Model", use_column_width=True)
    except FileNotFoundError:
        st.error(f"Confusion matrix image not found at `{CONFUSION_MATRIX_PATH}`. Please run `model_training.py` to generate model plots.")
    except Exception as e:
        st.error(f"Error loading confusion matrix image: {e}")

    # Display Feature Importances Image
    st.subheader("Feature Importances")
    try:
        feature_imp_img = Image.open(FEATURE_IMPORTANCES_PATH)
        st.image(feature_imp_img, caption="Feature Importances for RandomForest Model", use_column_width=True)
    except FileNotFoundError:
        st.error(f"Feature importances image not found at `{FEATURE_IMPORTANCES_PATH}`. Please run `model_training.py` to generate model plots.")
    except Exception as e:
        st.error(f"Error loading feature importances image: {e}")

# --- Tab 3: Raw Cleaned Data (snmdd_dataset_cleaned.csv) ---
with tab3:
    st.header("📋 Raw Cleaned Social Network Post Data")
    st.write("This table shows a sample of the cleaned individual social media posts before feature engineering was applied. This is the foundation of the analysis.")
    st.info("Displaying the entire raw dataset might consume significant memory and time for very large datasets. Showing only the first 1000 rows for performance.")

    @st.cache_data # Cache data loading for performance
    def load_cleaned_data(path):
        return pd.read_csv(path)

    try:
        cleaned_df = load_cleaned_data(CLEANED_DATA_PATH)
        st.dataframe(cleaned_df.head(1000)) # Show only first 1000 rows for performance
        st.write(f"Displaying first 1000 rows of {cleaned_df.shape[0]} total rows.")

        st.subheader("Basic Statistics of Cleaned Data")
        st.write(cleaned_df.describe())


    except FileNotFoundError:
        st.error(f"Cleaned dataset file not found at `{CLEANED_DATA_PATH}`. Please ensure your data cleaning process (e.g., `data_cleaning.py` or `main.py`) has been run.")
    except pd.errors.EmptyDataError:
        st.error(f"The file `{CLEANED_DATA_PATH}` is empty. Please check the data source.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading cleaned data: {e}")