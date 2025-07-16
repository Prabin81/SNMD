import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image # For loading images like confusion matrix

# --- Access Control ---
# Ensure user is logged in before rendering this page
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("🔒 Please log in as an admin from the home page to access this dashboard.")
    st.stop() # Stop execution if not logged in

script_dir = os.path.dirname(os.path.abspath(__file__))

base_project_dir = os.path.join(script_dir, "..", "..")

# --- Load Data ---

LABELED_DATA_PATH = os.path.join(script_dir, "..", "..", "outputs", "user_features_labeled.csv")
CLEANED_DATA_PATH = os.path.join(script_dir, "..", "..", "outputs", "snmdd_dataset_cleaned.csv")
MODEL_RESULTS_DIR = os.path.join(script_dir, "..", "..", "outputs", "ssl_results") # Based on image
# Assuming model_rf.pkl used by admin is from the top-level outputs
MODEL_RF_PATH = os.path.join(script_dir, "..", "..", "outputs", "model_rf.pkl")


st.title("📊 Admin Dashboard: Social Network Data Insights")
st.markdown("This dashboard provides an overview of the processed social network data and machine learning model performance.")

# --- Tabbed Navigation ---
tab1, tab2, tab3 = st.tabs(["Processed Data", "Model Performance", "Raw Cleaned Data"])

# --- Tab 1: Processed Data (user_features_labeled.csv) ---
with tab1:
    st.header("🔬 Processed User Features and Labels")
    st.write("This table shows the engineered features and generated risk labels for each user, used for training the main SNMD prediction model.")

    try:
        labeled_df = pd.read_csv(LABELED_DATA_PATH)
        st.dataframe(labeled_df)

        st.subheader("Summary Statistics of Labeled Features")
        st.write(labeled_df.describe())

        st.subheader("Distribution of Generated Risk Labels")
        fig, ax = plt.subplots()
        # Ensure labels are treated as categories for correct plotting
        label_counts = labeled_df['label'].value_counts().sort_index()
        # Map labels to more descriptive names if applicable (0: Low, 1: Moderate, 2: High)
        label_names = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
        label_counts.index = label_counts.index.map(label_names)

        label_counts.plot(kind='bar', ax=ax, color=['lightgreen', 'skyblue', 'lightcoral'])
        ax.set_title('Distribution of Generated Risk Labels Across Users')
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Number of Users')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig) # Close figure to free memory

        st.subheader("Correlation Matrix of Features")
        # Exclude non-numeric columns and the 'label' itself for correlation calculation
        numeric_cols = labeled_df.select_dtypes(include=np.number).columns.drop('label', errors='ignore')
        if not numeric_cols.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(labeled_df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Matrix of Engineered Features')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No numeric features found for correlation analysis.")


    except FileNotFoundError:
        st.error(f"Labeled data not found at `{LABELED_DATA_PATH}`. Please ensure `label_generation.py` has been run.")
    except Exception as e:
        st.error(f"Error loading or processing labeled data: {e}")

# --- Tab 2: Model Performance ---
with tab2:
    st.header("📈 Machine Learning Model Performance (RandomForest Classifier)")
    st.write("Evaluation metrics and visualizations for the RandomForest Classifier trained on the social network data features.")

    # Load Classification Report
    st.subheader("Classification Report")
    report_path = os.path.join(MODEL_RESULTS_DIR, "rf_classification_report.txt")
    try:
        with open(report_path, "r") as f:
            report_content = f.read()
        st.text(report_content)
    except FileNotFoundError:
        st.error(f"Classification report not found at `{report_path}`. Please run `model_training.py`.")
    except Exception as e:
        st.error(f"Error loading classification report: {e}")

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    confusion_matrix_path = os.path.join(MODEL_RESULTS_DIR, "confusion_matrix_rf.png")
    try:
        conf_matrix_img = Image.open(confusion_matrix_path)
        st.image(conf_matrix_img, caption="Confusion Matrix for RandomForest Model", use_column_width=True)
    except FileNotFoundError:
        st.error(f"Confusion matrix image not found at `{confusion_matrix_path}`. Please run `model_training.py`.")
    except Exception as e:
        st.error(f"Error loading confusion matrix image: {e}")

    # Display Feature Importances
    st.subheader("Feature Importances")
    feature_importances_path = os.path.join(MODEL_RESULTS_DIR, "feature_importances_rf.png")
    try:
        feature_imp_img = Image.open(feature_importances_path)
        st.image(feature_imp_img, caption="Feature Importances for RandomForest Model", use_column_width=True)
    except FileNotFoundError:
        st.error(f"Feature importances image not found at `{feature_importances_path}`. Please run `model_training.py`.")
    except Exception as e:
        st.error(f"Error loading feature importances image: {e}")

    # Optional: Display model details (e.g., if you want to inspect its parameters)
    # try:
    #     import pickle
    #     with open(MODEL_RF_PATH, "rb") as f:
    #         rf_model = pickle.load(f)
    #     st.subheader("Model Details")
    #     st.write(f"Model Type: {type(rf_model).__name__}")
    #     st.write("Model Parameters:")
    #     st.json(rf_model.get_params()) # Display model parameters
    # except FileNotFoundError:
    #     st.info(f"RandomForest model not found at `{MODEL_RF_PATH}`. Cannot display details.")
    # except Exception as e:
    #     st.error(f"Error loading RandomForest model for details: {e}")


# --- Tab 3: Raw Cleaned Data (snmdd_dataset_cleaned.csv) ---
with tab3:
    st.header("📋 Raw Cleaned Social Network Post Data")
    st.write("This table shows a sample of the cleaned individual social media posts before feature engineering was applied. This is the foundation of the analysis.")
    st.info("Displaying the entire raw dataset might consume significant memory and time for very large datasets. Showing only the first 1000 rows for performance.")

    try:
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
        st.dataframe(cleaned_df.head(1000)) # Show only first 1000 rows for performance
        st.write(f"Displaying first 1000 rows of {cleaned_df.shape[0]} total rows.")

        st.subheader("Basic Statistics of Cleaned Data")
        st.write(cleaned_df.describe())

        st.subheader("Example of Cleaned Text Content")
        for i, row in cleaned_df.head(5).iterrows():
            st.write(f"**Post {i+1}:** {row['cleaned_text']}")

    except FileNotFoundError:
        st.error(f"Cleaned dataset not found at `{CLEANED_DATA_PATH}`. Please ensure `main.py` (or `data_cleaning.py`) has been run.")
    except Exception as e:
        st.error(f"Error loading cleaned data: {e}")