import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Simulate user-level questionnaire data
# Create synthetic data for training model on 3 inputs: DSM-5, IAT, DASS
n_samples = 1000
np.random.seed(42)
data = {
    "dsm_score": np.random.randint(0, 6, size=n_samples),      # 0–5
    "iat_score": np.random.randint(5, 26, size=n_samples),     # 5–25
    "dass_score": np.random.randint(0, 16, size=n_samples),    # 0–15
    "label": np.random.randint(0, 2, size=n_samples)
}
df = pd.DataFrame(data)

# Split
X = df[["dsm_score", "iat_score", "dass_score"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained using 3 features and saved to model.pkl")
