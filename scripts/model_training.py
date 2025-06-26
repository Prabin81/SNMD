import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --- Load labeled dataset ---
data = pd.read_csv("SNMD/outputs/user_features_labeled.csv")

# --- Recommended features for modeling ---
recommended_features = [
    'avg_sentiment',
    'neg_post_ratio',
    'night_activity_ratio',
    'engagement_volatility',
    'avg_emotional_words',
    'avg_post_interval'
]

print("\n✅ Recommended features for modeling:")
for feat in recommended_features:
    print(f" - {feat}")

# --- Feature selection ---
X = data[recommended_features].fillna(0)
y = data['label']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model training ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Evaluation ---
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Classification Report ===")
print(report)
print("\n✅ Accuracy Score:", accuracy)

# --- Confusion Matrix plot ---
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not At Risk', 'At Risk'],
            yticklabels=['Not At Risk', 'At Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()

# --- Save Outputs ---
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Save model
with open(os.path.join(output_dir, "model_rf.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save classification report
with open(os.path.join(output_dir, "rf_classification_report.txt"), "w") as f:
    f.write(report)
    f.write(f"\nAccuracy: {accuracy:.4f}")

# Save confusion matrix plot
plt.savefig(os.path.join(output_dir, "confusion_matrix_rf.png"))
plt.show()

print(f"\n✅ Model and results saved in: {output_dir}/")
