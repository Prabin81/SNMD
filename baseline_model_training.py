import pandas as pd
import numpy as np
import os
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("Starting baseline model training process...")

# --- Dynamic Path Handling ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input path: labeled features (Correctly points to top-level outputs)
input_path = os.path.join(script_dir, "outputs", "user_features_labeled.csv") # Or user_features_ssl_labeled.csv

# Output directory for baseline results (Correctly points to top-level outputs)
output_dir = os.path.join(script_dir, "outputs", "baseline_results")
os.makedirs(output_dir, exist_ok=True)

# --- Features for Baseline Model ---
# For a true baseline, we can use the same prediction features as our main model
# or even a simpler set to show minimum performance.
BASELINE_FEATURES = [
    'avg_likes',
    'avg_comments',
    'avg_engagement',
    'total_posts',
]

# Load data
data = None
try:
    data = pd.read_csv(input_path)
    if 'reciprocity' in data.columns and 'reciprocity' not in BASELINE_FEATURES:
        BASELINE_FEATURES.append('reciprocity')
    print(f"✅ Loaded labeled data from {input_path}")
except FileNotFoundError:
    print(f"❌ File not found at {input_path}")
    print("💡 Please run label_generation.py first to generate the labeled dataset.")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred while loading data: {e}")
    exit()

if data is None:
    exit()

# Prepare X and y
missing_feats_baseline = [f for f in BASELINE_FEATURES if f not in data.columns]
if missing_feats_baseline:
    print(f"❌ Error: The following baseline features are missing from the dataset: {missing_feats_baseline}")
    print("Please check 'feature_engineer.py' and 'label_generation.py' outputs.")
    exit()

X = data[BASELINE_FEATURES].fillna(0) # Fill NaNs for selected features
y = data['label']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Class distribution in target:\n{y.value_counts()}")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n✅ Data split into training and testing sets.")

# --- Train Baseline Model ---
# DummyClassifier makes predictions that ignore the input features.
# 'stratified': Generates predictions by respecting the training set’s class distribution.
# 'most_frequent': Always predicts the most frequent label in the training set.
model = DummyClassifier(strategy='most_frequent', random_state=42)
# model = DummyClassifier(strategy='stratified', random_state=42) # Another option

print("\n🚀 Training DummyClassifier (Baseline Model)...")
model.fit(X_train, y_train)
print("✅ Baseline model training complete.")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Baseline Model Classification Report ===")
print(report)
print(f"\n✅ Baseline Accuracy Score: {accuracy:.4f}")

# Save results
report_path = os.path.join(output_dir, "baseline_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
    f.write(f"\nAccuracy: {accuracy:.4f}")
print(f"✅ Baseline classification report saved to: {report_path}")

print("\nBaseline model process complete.")