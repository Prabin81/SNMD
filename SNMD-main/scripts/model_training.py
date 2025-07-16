import os
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import shap # Added SHAP import
import lime # Added LIME import
import lime.lime_tabular # Specific LIME module for tabular data

print("Starting model training and evaluation process...")

# Get the directory of the current script (model_training.py)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the root project directory (E:\SNMD-MAIN\)
root_project_dir = os.path.join(script_dir, "..", "..")

# --- Configuration ---
DATA_PATH = os.path.join(root_project_dir, "outputs", "user_features_labeled.csv")
OUTPUT_DIR_FOR_RESULTS = os.path.join(root_project_dir, "outputs", "rf_results") # Changed output dir for RF
os.makedirs(OUTPUT_DIR_FOR_RESULTS, exist_ok=True) # Create if it doesn't exist
MODEL_RF_SAVE_PATH = os.path.join(root_project_dir, "outputs", "model_rf.pkl")

# --- CRITICAL CHANGE: FEATURES FOR PREDICTION ---
# Define core features that should always be present.
# 'reciprocity' will be added dynamically if available in the dataset.
PREDICTION_FEATURES_CORE = [
    'avg_likes',
    'avg_comments',
    'avg_engagement',
    'total_posts',
    'avg_sentiment', # Assuming these are available from feature engineering
    'neg_post_ratio',
    'avg_post_interval',
    'avg_emotional_words',
    'night_activity_ratio',
    'engagement_volatility'
]
# Initialize PREDICTION_FEATURES with core features.
PREDICTION_FEATURES = list(PREDICTION_FEATURES_CORE)


# --- Noise Configuration (Keep for simulating real-world imperfections if needed) ---
NOISE_MEAN = 0
# INCREASED NOISE STANDARD DEVIATION FURTHER TO REDUCE ACCURACY AS REQUESTED
NOISE_STD_DEV = 0.1 # Increased from 0.05 to 0.1. Adjust further if needed.


# --- Load Labeled Dataset ---
try:
    data = pd.read_csv(DATA_PATH)
    # Check if 'reciprocity' column exists and add it to PREDICTION_FEATURES if it does
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
for feat in final_prediction_features: # Use final_prediction_features for printing
    print(f" - {feat}")


X = data[final_prediction_features].fillna(0) # Fill NaN with 0, consider other strategies like mean/median
y = data['label']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Class distribution in target:\n{y.value_counts()}")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n✅ Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train distribution:\n{y_train.value_counts()}")
print(f"y_test distribution:\n{y_test.value_counts()}")

# --- Introduce Noise to Training Data (Optional, for fine-tuning accuracy) ---
print(f"\n✨ Introducing Gaussian noise (mean={NOISE_MEAN}, std_dev={NOISE_STD_DEV}) to training features...")
noise = np.random.normal(NOISE_MEAN, NOISE_STD_DEV, X_train.shape)
X_train_noisy = X_train + noise
print("✅ Noise added to X_train.")

# --- Model Training (with Hyperparameter Tuning) ---
print("\n🚀 Training RandomForestClassifier with GridSearchCV...")

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150], # Number of trees in the forest
    'max_depth': [5, 10, 15, None], # Maximum depth of the tree
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'class_weight': ['balanced'] # Keep balanced for potentially imbalanced classes
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5, # 5-fold cross-validation during tuning
    scoring='accuracy',
    n_jobs=-1, # Use all available cores
    verbose=1
)

grid_search.fit(X_train_noisy, y_train)

model = grid_search.best_estimator_ # Get the best model from grid search
print(f"✅ Best RandomForestClassifier parameters found: {grid_search.best_params_}")
print("✅ Model training complete.")


# --- Predictions ---
print("Generating predictions on the test set (clean data)...")
y_pred = model.predict(X_test) # Predict on clean test data
print("✅ Predictions generated.")

# --- Evaluation ---
print("\n=== Model Evaluation ===")
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Classification Report ===")
print(report)
print(f"\n✅ Accuracy Score on Test Set: {accuracy:.4f}")

# --- Interpretation of Accuracy ---
# Adjusted logic to reflect the user's desired accuracy range (85-90%)



# --- Cross-Validation (Robust evaluation on the full dataset) ---
print("\n--- Performing 5-Fold Cross-Validation on FULL dataset (with noise per fold) ---")
cv_accuracies = []
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_fold_train, X_fold_val = X.iloc[train_index], X.iloc[val_index]
    y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]

    # Apply noise to the training set of the current fold
    fold_noise = np.random.normal(NOISE_MEAN, NOISE_STD_DEV, X_fold_train.shape)
    X_fold_train_noisy = X_fold_train + fold_noise

    # Use best parameters found by GridSearchCV for CV model
    cv_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    cv_model.fit(X_fold_train_noisy, y_fold_train)
    fold_pred = cv_model.predict(X_fold_val)
    fold_accuracy = accuracy_score(y_fold_val, fold_pred)
    cv_accuracies.append(fold_accuracy)
    print(f"Fold {fold+1} Accuracy: {fold_accuracy:.4f}")

cv_scores_mean = np.mean(cv_accuracies)
cv_scores_std = np.std(cv_accuracies)

print(f"Cross-validation accuracy scores (5-folds): {cv_accuracies}")
print(f"Mean CV Accuracy: {cv_scores_mean:.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores_std:.4f}")

# --- Feature Importance (based on the best model from GridSearchCV) ---
print("\n=== Feature Importances (from Best Model) ===")
if not model.feature_importances_.size == 0:
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_importances)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importances.values, y=feature_importances.index)
    plt.title('Feature Importances (Best Model)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_FOR_RESULTS, "feature_importances_rf.png")) # Save to rf_results
else:
    print("Could not compute feature importances.")


# --- Confusion Matrix Plot ---
print("\nGenerating Confusion Matrix plot...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
# Dynamically determine class labels based on unique values in y_test
class_labels = [f"Class {c}" for c in sorted(np.unique(y_test))] # Default if not known
if len(np.unique(y_test)) == 3: # Assuming 3 classes (0, 1, 2)
    class_labels = ['Low Risk', 'Moderate Risk', 'High Risk'] #
elif len(np.unique(y_test)) == 2: # Assuming 2 classes (0, 1)
    class_labels = ['Low Risk', 'High Risk'] #
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, # Adjusted for dynamic classes
            yticklabels=class_labels) # Adjusted for dynamic classes
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR_FOR_RESULTS, "confusion_matrix_rf.png")) # Save to rf_results
plt.close('all') # Close plot to prevent it from showing immediately


# --- SHAP Feature Importance (for Random Forest) ---
print("\n=== Generating SHAP explanations ===")
try:
    # Use shap.TreeExplainer for tree-based models, it's very efficient
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot summary (e.g., beeswarm or bar)
    plt.figure(figsize=(10, 6))
    # If your model predicts multiple classes, shap_values will be a list of arrays.
    # We typically plot for a specific class or take the absolute mean.
    # Assuming binary classification for simplicity (shap_values[1] for positive class)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap.summary_plot(shap_values[1], X_test, feature_names=final_prediction_features, show=False)
    else: # For single-output models or if only one array is returned
        shap.summary_plot(shap_values, X_test, feature_names=final_prediction_features, show=False)
    
    plt.title('SHAP Feature Importance (Random Forest)')
    plt.tight_layout()
    shap_plot_path = os.path.join(OUTPUT_DIR_FOR_RESULTS, "shap_summary_plot_rf.png")
    plt.savefig(shap_plot_path)
    print(f"✅ SHAP summary plot saved to: {shap_plot_path}")
    plt.close('all') # Close plot
except Exception as e:
    print(f"❌ Error generating SHAP plot: {e}")


# --- LIME Explanation (for Random Forest) ---
print("\n=== Generating LIME explanation for a sample prediction ===")
try:
    # Create a LIME explainer
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values, # LIME uses training data for feature distribution
        feature_names=final_prediction_features,
        class_names=[str(c) for c in model.classes_], # Ensure class names are strings
        mode='classification'
    )

    # Choose a random instance from the test set to explain
    np.random.seed(42) # For reproducibility
    idx = np.random.randint(0, len(X_test))
    instance = X_test.iloc[idx].values
    true_label = y_test.iloc[idx]
    predicted_label = model.predict(instance.reshape(1, -1))[0]

    print(f"Explaining instance {idx}: True Label = {true_label}, Predicted Label = {predicted_label}")

    # Explain the prediction
    explanation = explainer_lime.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba, # LIME needs predict_proba for classification
        num_features=len(final_prediction_features) # Explain all features
    )

    # Save the LIME explanation plot as HTML (CORRECTED LINE HERE)
    lime_plot_path = os.path.join(OUTPUT_DIR_FOR_RESULTS, f"lime_explanation_rf_instance_{idx}.html")
    # Use save_to_file() if available, otherwise get HTML string and write it.
    try:
        explanation.save_to_file(lime_plot_path) # Preferred method
    except AttributeError:
        # Fallback if save_to_file is not available, typically means older LIME version
        with open(lime_plot_path, 'w') as f:
            f.write(explanation.as_html())
    
    print(f"✅ LIME explanation for instance {idx} saved to: {lime_plot_path}")
    print("To view, open the HTML file in a web browser.")
except Exception as e:
    print(f"❌ Error generating LIME explanation: {e}")


# --- Save Outputs ---
print(f"\nSaving model and results...")

# Save main model (RandomForestClassifier) to E:\SNMD-MAIN\outputs\
with open(MODEL_RF_SAVE_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Main RandomForest Model saved to: {MODEL_RF_SAVE_PATH}")

# Save classification report to E:\SNMD-MAIN\outputs\rf_results\
report_path = os.path.join(OUTPUT_DIR_FOR_RESULTS, "rf_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
    f.write(f"\n\nAccuracy on Test Set: {accuracy:.4f}")
    f.write(f"\nMean CV Accuracy: {cv_scores_mean:.4f}")
    f.write(f"\nStd Dev CV Accuracy: {cv_scores_std:.4f}")
    f.write(f"\n\nBest Hyperparameters: {grid_search.best_params_}")
print(f"✅ Classification report saved to: {report_path}")


print(f"\n🥳 Model training and evaluation complete. Results in: {root_project_dir}/outputs/ and {OUTPUT_DIR_FOR_RESULTS}/")