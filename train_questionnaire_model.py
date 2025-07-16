import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("Starting questionnaire-based model training process...")

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Output directory for questionnaire model (SNMD-main/models/questionnaire_model/)
OUTPUT_DIR = os.path.join(script_dir, "models", "questionnaire_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulate user-level questionnaire data
# This data is INDEPENDENT of the social media data.
# It represents an an alternative way to detect risk, possibly based on clinical scores.
n_samples = 1000
np.random.seed(42)
# Ensure labels are binary (0 or 1) for a classification task
dsm_score = np.random.randint(0, 6, size=n_samples)      # 0–5
iat_score = np.random.randint(5, 26, size=n_samples)     # 5–25
dass_score = np.random.randint(0, 16, size=n_samples)    # 0–15

# Generate labels that are somewhat correlated with scores, but not perfectly binary initially
# Threshold at 0.5 to make it binary 0 or 1
raw_label_score = (
    (np.random.rand(n_samples) * 0.5) +   # Add some randomness
    (dsm_score * 0.1) +
    (iat_score * 0.02) + # Reduced influence to make it less perfectly predictable
    (dass_score * 0.03)
)
label = (raw_label_score > raw_label_score.mean()).astype(int) # Simple binary split

data = {
    "dsm_score": dsm_score,
    "iat_score": iat_score,
    "dass_score": dass_score,
    "label": label
}
df = pd.DataFrame(data)

print(f"Generated synthetic questionnaire dataset shape: {df.shape}")
print(f"Synthetic label distribution:\n{df['label'].value_counts()}")

# Split
X = df[["dsm_score", "iat_score", "dass_score"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n✅ Synthetic data split into training and testing sets.")

# Train Logistic Regression model
print("\n🚀 Training Logistic Regression model on synthetic questionnaire data...")
model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is often good for small datasets
model.fit(X_train, y_train)
print("✅ Logistic Regression model training complete.")

# Evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Logistic Regression Model Classification Report (Questionnaire Data) ===")
print(report)
print(f"\n✅ Accuracy Score (Questionnaire Data): {accuracy:.4f}")


# Save model and report
model_path = os.path.join(OUTPUT_DIR, "questionnaire_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Model trained using 3 features and saved to {model_path}")

report_path = os.path.join(OUTPUT_DIR, "questionnaire_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
    f.write(f"\nAccuracy: {accuracy:.4f}")
print(f"✅ Classification report saved to: {report_path}")

print("\nQuestionnaire-based model process complete.")