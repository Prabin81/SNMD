import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load labeled dataset - use raw string or forward slashes
data = pd.read_csv("outputs/user_features_labeled.csv")

# Define features and target
features = [
    'avg_likes',
    'avg_comments',
    'avg_engagement',
    'avg_sentiment',
    'neg_post_ratio',
    'total_posts',
    'avg_post_interval'
    # 'parasociality_ratio'  # Remove if not available
]

X = data[features].fillna(0)
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not At Risk', 'At Risk'],
            yticklabels=['Not At Risk', 'At Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()

# Ensure output directory exists before saving
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()
