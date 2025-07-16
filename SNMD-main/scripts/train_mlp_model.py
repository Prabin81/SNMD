import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap # Added SHAP import

print("Starting MLP model training and evaluation process...")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_project_dir = os.path.join(script_dir, "..") # SNMD-MAIN/

DATA_PATH = os.path.join(root_project_dir, "outputs", "user_features_labeled.csv") # Or user_features_ssl_labeled.csv
OUTPUT_DIR_FOR_RESULTS = os.path.join(root_project_dir, "outputs", "mlp_results")
os.makedirs(OUTPUT_DIR_FOR_RESULTS, exist_ok=True)
MODEL_MLP_SAVE_PATH = os.path.join(root_project_dir, "outputs", "model_mlp.pkl") # Save MLP model

# --- CRITICAL CHANGE: FEATURES FOR PREDICTION ---
# Define core features that should always be present.
# 'reciprocity' will be added dynamically if available in the dataset.
PREDICTION_FEATURES_CORE = [
    'avg_likes',
    'avg_comments',
    'avg_engagement',
    'total_posts',
    'avg_sentiment',
    'neg_post_ratio',
    'avg_post_interval',
    'avg_emotional_words',
    'night_activity_ratio',
    'engagement_volatility'
]
# Initialize PREDICTION_FEATURES with core features.
PREDICTION_FEATURES = list(PREDICTION_FEATURES_CORE)


try:
    data = pd.read_csv(DATA_PATH)
    # Dynamically add 'reciprocity' ONLY if it exists in the loaded data
    if 'reciprocity' in data.columns and 'reciprocity' not in PREDICTION_FEATURES:
        PREDICTION_FEATURES.append('reciprocity')

    print(f"✅ Successfully loaded data from: {DATA_PATH}")
    print(f"Dataset shape: {data.shape}")
except FileNotFoundError:
    print(f"❌ Error: Labeled dataset not found at {DATA_PATH}. Please run label_generation.py first.")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred while loading data: {e}")
    exit()

# Filter PREDICTION_FEATURES to only include those actually present in the loaded data.
# This ensures no KeyError even if some features (like 'reciprocity') were skipped.
final_prediction_features = [f for f in PREDICTION_FEATURES if f in data.columns]
missing_from_expected = [f for f in PREDICTION_FEATURES if f not in data.columns]

if missing_from_expected:
    print(f"⚠️ Warning: The following expected features are missing from the dataset and will be skipped: {missing_from_expected}")
    print("Please check your feature engineering process if these are expected to be present.")

print("\n✅ Features selected for modeling:")
for feat in final_prediction_features:
    print(f" - {feat}")

X = data[final_prediction_features].fillna(0) # Use the filtered list of features
y = data['label']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Class distribution in target:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🚀 Training MLPClassifier with GridSearchCV...")
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)], # Example configurations
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'], # Explicitly include 'sgd' to satisfy the requirement
    'alpha': [0.0001, 0.001, 0.01], # L2 regularization term
    'learning_rate_init': [0.001, 0.01]
}

grid_search_mlp = GridSearchCV(
    estimator=MLPClassifier(max_iter=500, random_state=42), # Increased max_iter for convergence
    param_grid=param_grid,
    cv=3, # Reduced CV folds for faster tuning during development
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_mlp.fit(X_train, y_train)

mlp_model = grid_search_mlp.best_estimator_
print(f"✅ Best MLPClassifier parameters found: {grid_search_mlp.best_params_}")
print("✅ MLP model training complete.")

y_pred_mlp = mlp_model.predict(X_test)

report_mlp = classification_report(y_test, y_pred_mlp)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print("\n=== MLP Model Evaluation ===")
print(report_mlp)
print(f"\n✅ Accuracy Score on Test Set: {accuracy_mlp:.4f}")

# Confusion Matrix Plot
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(6, 4))
class_labels = [f"Class {c}" for c in sorted(np.unique(y_test))]
if len(np.unique(y_test)) == 3:
    class_labels = ['Low Risk', 'Moderate Risk', 'High Risk']
elif len(np.unique(y_test)) == 2:
    class_labels = ['Low Risk', 'High Risk']
sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('MLP Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR_FOR_RESULTS, "confusion_matrix_mlp.png"))
plt.close('all')


# --- SHAP Feature Importance (for MLP) ---
print("\n=== Generating SHAP explanations for MLP ===")
try:
    # For non-tree models, shap.KernelExplainer is generally used.
    # It requires a background dataset for integration calculation.
    # Using a small random sample of the training data as background.
    background_data_sample_size = min(100, X_train.shape[0]) # Use a smaller sample for speed
    background_data = X_train.iloc[np.random.choice(X_train.shape[0], background_data_sample_size, replace=False)]

    explainer_mlp = shap.KernelExplainer(mlp_model.predict_proba, background_data)

    # Explain a sample of test data. KernelExplainer can be slow for large datasets.
    test_data_sample_size = min(500, X_test.shape[0]) # Explain up to 500 instances
    X_test_sample = X_test.sample(test_data_sample_size, random_state=42) if test_data_sample_size < X_test.shape[0] else X_test

    shap_values_mlp = explainer_mlp.shap_values(X_test_sample)

    plt.figure(figsize=(10, 6))
    # If your model predicts multiple classes, shap_values_mlp will be a list of arrays.
    # Plotting for the positive class (assuming binary classification, usually index 1)
    if isinstance(shap_values_mlp, list) and len(shap_values_mlp) > 1:
        shap.summary_plot(shap_values_mlp[1], X_test_sample, feature_names=final_prediction_features, show=False)
    else: # For single-output models or if only one array is returned
        shap.summary_plot(shap_values_mlp, X_test_sample, feature_names=final_prediction_features, show=False)

    plt.title('SHAP Feature Importance (MLP)')
    plt.tight_layout()
    shap_plot_path_mlp = os.path.join(OUTPUT_DIR_FOR_RESULTS, "shap_summary_plot_mlp.png")
    plt.savefig(shap_plot_path_mlp)
    print(f"✅ SHAP summary plot for MLP saved to: {shap_plot_path_mlp}")
    plt.close('all') # Close plot to prevent it from showing immediately

except Exception as e:
    print(f"❌ Error generating SHAP explanation for MLP: {e}")


# Save model and report
with open(MODEL_MLP_SAVE_PATH, "wb") as f:
    pickle.dump(mlp_model, f)
print(f"✅ MLP Model saved to: {MODEL_MLP_SAVE_PATH}")

report_path_mlp = os.path.join(OUTPUT_DIR_FOR_RESULTS, "mlp_classification_report.txt")
with open(report_path_mlp, "w") as f:
    f.write(report_mlp)
    f.write(f"\n\nAccuracy on Test Set: {accuracy_mlp:.4f}")
    f.write(f"\nBest Hyperparameters: {grid_search_mlp.best_params_}")
print(f"✅ MLP classification report saved to: {report_path_mlp}")

print(f"\n🥳 MLP model training and evaluation complete. Results in: {OUTPUT_DIR_FOR_RESULTS}/")