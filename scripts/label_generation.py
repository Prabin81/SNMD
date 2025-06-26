import pandas as pd
import os

# Load user features
user_features = pd.read_csv("user_features_final.csv")

# Generate labels heuristically
def generate_ordinal_label(row):
    """Multi-dimensional risk assessment with ordinal labels"""
    risk_score = 0

    # Sentiment (30% weight)
    if row['avg_sentiment'] < -0.5:
        risk_score += 0.3
    elif row['avg_sentiment'] < -0.2:
        risk_score += 0.15

    # Engagement patterns (25% weight)
    if row['avg_engagement'] < 5:
        risk_score += 0.25
    elif row.get('engagement_volatility', 0) > 15:
        risk_score += 0.15

    # Temporal patterns (25% weight)
    if row.get('night_activity_ratio', 0) > 0.4:
        risk_score += 0.25
    if row.get('avg_post_interval', float('inf')) < 1800:
        risk_score += 0.1

    # Emotional content (20% weight)
    if row.get('avg_emotional_words', 0) > 2:
        risk_score += 0.2

    # Ordinal classification
    if risk_score < 0.4:
        return 0  # Low Risk
    elif risk_score < 0.7:
        return 1  # Moderate Risk
    else:
        return 2  # High Risk

# Apply the label
user_features['label'] = user_features.apply(generate_ordinal_label, axis=1)

# Create outputs directory if it doesn't exist
output_dir = r"F:\SNMD\SNMD\outputs"
os.makedirs(output_dir, exist_ok=True)

# Save to file with proper filename
output_path = os.path.join(output_dir, "user_features_labeled.csv")
user_features.to_csv(output_path, index=False)
print(f"✅ Ordinal labels generated and saved at {output_path}")