import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load labeled dataset 
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "output", "user_features_labeled.csv")

try:
    data = pd.read_csv(input_path)
    print(f"✅ Loaded data from {input_path}")
except FileNotFoundError:
    print("❌ Please generate labeled dataset first using label_generation.py")
    exit()

# Prepare features and labels 
# Select only numeric columns for features
X = data.select_dtypes(include=[np.number])

y = data['label']

# Simulate semi-supervised scenario 
# Mask 80% of labels as -1 (unlabeled)
rng = np.random.RandomState(42)
mask = rng.rand(len(y)) < 0.8
y_unlabeled = y.copy()
y_unlabeled[mask] = -1  # unlabeled

# Train Label Spreading model 
model = LabelSpreading(kernel='rbf', alpha=0.2)
model.fit(X, y_unlabeled)

#  Get predictions for all data 
predicted = model.transduction_

#  Evaluation 
print("\n📊 Classification Report (on all data):")
print(classification_report(y, predicted))

print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y, predicted))

# Save predictions 
data['ssl_label'] = predicted
output_path = os.path.join(script_dir, "output", "user_features_ssl.csv")
try:
    data.to_csv(output_path, index=False)
    print(f"\n✅ SSL labeled data saved to: {output_path}")
except Exception as e:
    print(f"❌ Failed to save SSL labeled data: {e}")
