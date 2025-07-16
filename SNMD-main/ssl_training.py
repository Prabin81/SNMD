import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.preprocessing import StandardScaler

print("Starting Semi-Supervised Learning process...")

# --- Dynamic Path Handling ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input path: labeled features from label_generation.py (Correctly points to top-level outputs)
input_path = os.path.join(script_dir, "outputs", "user_features_labeled.csv")

# Output directory for SSL results (Correctly points to top-level outputs)
output_dir = os.path.join(script_dir, "outputs", "ssl_results")
os.makedirs(output_dir, exist_ok=True) # Create output directory for SSL results

print(f"Attempting to load labeled data from: {input_path}") # Debugging print

# --- Features for SSL ---
# It's generally good to use ALL available relevant features for SSL as it learns manifold.
# This should include features used in labeling AND features used for prediction.
ALL_ENGINEERED_FEATURES = [
    'avg_likes', 'avg_comments', 'avg_engagement', 'avg_sentiment',
    'neg_post_ratio', 'total_posts', 'avg_post_interval',
    'avg_emotional_words', 'night_activity_ratio', 'engagement_volatility'
]

# Load labeled dataset (Adding more robust error handling and feature check)
data = None # Initialize data to None
try:
    data = pd.read_csv(input_path)
    # Check if 'reciprocity' column exists and add it to ALL_ENGINEERED_FEATURES if it does
    if 'reciprocity' in data.columns and 'reciprocity' not in ALL_ENGINEERED_FEATURES:
        ALL_ENGINEERED_FEATURES.append('reciprocity')

    print(f"✅ Loaded data from {input_path}")
except FileNotFoundError:
    print(f"❌ Error: Labeled dataset not found at {input_path}")
    print("💡 Please ensure 'label_generation.py' has been run successfully and the file exists at this exact path.")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred while loading data: {e}")
    exit()

# If data loading failed, exit
if data is None:
    exit()

# Prepare features and labels
# Ensure all selected features exist and handle NaNs
missing_feats_ssl = [f for f in ALL_ENGINEERED_FEATURES if f not in data.columns]
if missing_feats_ssl:
    print(f"❌ Error: The following SSL features are missing from the dataset: {missing_feats_ssl}")
    print("Please check 'feature_engineer.py' and 'label_generation.py' outputs.")
    exit()

X = data[ALL_ENGINEERED_FEATURES].fillna(0) # Fill NaNs for selected features
y = data['label']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Class distribution in target:\n{y.value_counts()}")

# Simulate semi-supervised scenario (masking labels already done in label_generation.py)
# Ensure y_unlabeled is used for training, where -1 are unlabeled data points
y_unlabeled = y.copy()
print(f"Distribution of labels for SSL training:\n{pd.Series(y_unlabeled).value_counts()}")

# Train Label Spreading model
print("\n🚀 Training Label Spreading model...")
# Consider scaling X if features have very different ranges, as RBF kernel is distance-based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Scale features for RBF kernel

model = LabelSpreading(kernel='rbf', gamma=20, alpha=0.2, max_iter=30) # Increased gamma, max_iter for potentially better performance
model.fit(X_scaled, y_unlabeled) # Train on scaled features
print("✅ Label Spreading model training complete.")

# Get predictions for all data (transduction)
predicted = model.transduction_

# Evaluation (only if 'y' contains original labels for evaluation)
# Filter out -1 labels from both y and predicted for evaluation
valid_indices = y != -1
if valid_indices.sum() > 0:
    print("\n📊 Classification Report (on all data, comparing true vs transduced labels for known points):")
    print(classification_report(y[valid_indices], predicted[valid_indices]))

    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(y[valid_indices], predicted[valid_indices]))
else:
    print("No original labeled data points available for evaluation.")

# Save predictions
data['ssl_label'] = predicted
output_path = os.path.join(output_dir, "user_features_ssl_labeled.csv")
try:
    data.to_csv(output_path, index=False)
    print(f"\n✅ SSL labeled data saved to: {output_path}")
except Exception as e:
    print(f"❌ Failed to save SSL labeled data: {e}")

print("Semi-Supervised Learning process complete.")