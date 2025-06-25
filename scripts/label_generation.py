import pandas as pd

# Load user features
user_features = pd.read_csv("user_features_final.csv")

# Generate labels heuristically
def generate_labels(row):
    if row['avg_sentiment'] < -0.2 and row['avg_engagement'] < 10:
        return 1  # At risk
    else:
        return 0  # Not at risk

user_features['label'] = user_features.apply(generate_labels, axis=1)

# Save labeled dataset
user_features.to_csv("outputs/user_features_labeled.csv", index=False)
print("✅ Labels generated and saved at outputs/user_features_labeled.csv")
