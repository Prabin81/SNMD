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

print("Starting model training and evaluation process...")

# Get the directory of the current script (model_training.py)
# This will be E:\SNMD-MAIN\SNMD-main\scripts\
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the root project directory (E:\SNMD-MAIN\)
# From scripts, go up one level (to SNMD-main), then up another level (to SNMD-MAIN)
root_project_dir = os.path.join(script_dir, "..", "..")

# --- Configuration ---
# Data input path: E:\SNMD-MAIN\outputs\user_features_labeled.csv
DATA_PATH = os.path.join(root_project_dir, "outputs", "user_features_labeled.csv")

# Output directory for model results (e.g., reports, plots)
# It seems your Admin_Dashboard.py expects these in 'ssl_results' under 'outputs'
OUTPUT_DIR_FOR_RESULTS = os.path.join(root_project_dir, "outputs", "ssl_results")
os.makedirs(OUTPUT_DIR_FOR_RESULTS, exist_ok=True) # Create if it doesn't exist

# Output path for the trained main RandomForest model
# Admin_Dashboard.py expects model_rf.pkl directly in E:\SNMD-MAIN\outputs\
MODEL_RF_SAVE_PATH = os.path.join(root_project_dir, "outputs", "model_rf.pkl")

# --- CRITICAL CHANGE: FEATURES FOR PREDICTION ---
# These features should NOT be the ones directly summed/used in your generate_ordinal_label heuristic.
# We're using general engagement metrics and post counts, which might correlate with risk
# but are not directly the 'components' of the heuristic score.
PREDICTION_FEATURES = [
    'avg_likes',
    'avg_comments',
    'avg_engagement',
    'total_posts',
    # 'reciprocity' # Will be added dynamically if available
]

# --- Noise Configuration (Keep for simulating real-world imperfections if needed) ---
NOISE_MEAN = 0
NOISE_STD_DEV = 0.005 # Reduced noise, as feature separation is the primary fix. Adjust as needed.

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

print("\n✅ Features selected for modeling (NOT directly used as components in label generation heuristic):")
for feat in PREDICTION_FEATURES:
    print(f" - {feat}")

# --- Feature Selection and Handling Missing Values ---
# Ensure all selected features exist and handle NaNs
missing_feats = [f for f in PREDICTION_FEATURES if f not in data.columns]
if missing_feats:
    print(f"❌ Error: The following prediction features are missing from the dataset: {missing_feats}")
    print("Please check 'feature_engineer.py' and 'label_generation.py' outputs.")
    exit()

X = data[PREDICTION_FEATURES].fillna(0) # Fill NaN with 0, consider other strategies like mean/median
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
if accuracy >= 0.90: # Adjust threshold for "very high" based on expectation after fix
    print("\n--- NOTE: HIGH ACCURACY DETECTED! ---")
    print("While data leakage from labeling has been addressed by feature separation,")
    print("a very high accuracy might still indicate the problem is relatively easy to learn,")
    print("or there might still be subtle indirect correlations between prediction features and label generation.")
    print("Consider further investigation or more diverse feature sets, or refine your heuristic.")
elif accuracy < 0.70:
    print("\n--- NOTE: LOW ACCURACY DETECTED! ---")
    print("Accuracy is relatively low. Consider:")
    print("1. Re-evaluating the 'PREDICTION_FEATURES' for stronger signals.")
    print("2. Adjusting RandomForestClassifier hyperparameters (more GridSearchCV iterations/ranges).")
    print("3. Re-evaluating the heuristic in label_generation.py if this accuracy is lower than desired.")
    print("--------------------------------------------------")


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
    plt.savefig(os.path.join(OUTPUT_DIR_FOR_RESULTS, "feature_importances_rf.png")) # Save to ssl_results
    #plt.show() # Commented out for automated runs
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

# --- Save Outputs ---
print(f"\nSaving model and results...")

# Save main model (RandomForestClassifier) to E:\SNMD-MAIN\outputs\
with open(MODEL_RF_SAVE_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Main RandomForest Model saved to: {MODEL_RF_SAVE_PATH}")

# Save classification report to E:\SNMD-MAIN\outputs\ssl_results\
report_path = os.path.join(OUTPUT_DIR_FOR_RESULTS, "rf_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
    f.write(f"\n\nAccuracy on Test Set: {accuracy:.4f}")
    f.write(f"\nMean CV Accuracy: {cv_scores_mean:.4f}")
    f.write(f"\nStd Dev CV Accuracy: {cv_scores_std:.4f}")
    f.write(f"\n\nBest Hyperparameters: {grid_search.best_params_}")
print(f"✅ Classification report saved to: {report_path}")

# Save confusion matrix plot to E:\SNMD-MAIN\outputs\ssl_results\
confusion_matrix_path = os.path.join(OUTPUT_DIR_FOR_RESULTS, "confusion_matrix_rf.png")
plt.savefig(confusion_matrix_path)
print(f"✅ Confusion Matrix plot saved to: {confusion_matrix_path}")

plt.close('all') # Close all plots to prevent them from showing immediately

print(f"\n🥳 Model training and evaluation complete. Results in: {root_project_dir}/outputs/ and {OUTPUT_DIR_FOR_RESULTS}/")