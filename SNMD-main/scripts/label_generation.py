import pandas as pd
import os

# --- Step 1: Dynamic Path Handling ---
# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input path (user features generated from feature_engineer.py)
input_path = os.path.join(script_dir, "..", "output", "user_features_final.csv")
output_dir = os.path.join(script_dir, "..", "output")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Step 2: Load Data ---
try:
    user_features = pd.read_csv(input_path)
    print(f"✅ Successfully loaded data from {input_path}")
except FileNotFoundError:
    print(f"❌ Error: File not found at {input_path}")
    print("💡 Please run feature_engineer.py first to generate the input file.")
    exit()

# --- Step 3: Heuristic Label Generation ---
def generate_ordinal_label(row):
    """Multi-dimensional risk assessment with ordinal labels"""
    risk_score = 0

    # Sentiment (30%)
    if row['avg_sentiment'] < -0.5:
        risk_score += 0.3
    elif row['avg_sentiment'] < -0.2:
        risk_score += 0.15

    # Engagement (25%)
    if row['avg_engagement'] < 5:
        risk_score += 0.25
    elif row.get('engagement_volatility', 0) > 15:
        risk_score += 0.15

    # Temporal patterns (25%)
    if row.get('night_activity_ratio', 0) > 0.4:
        risk_score += 0.25
    if row.get('avg_post_interval', float('inf')) < 1800:
        risk_score += 0.1

    # Emotional content (20%)
    if row.get('avg_emotional_words', 0) > 2:
        risk_score += 0.2

    # Label assignment
    if risk_score < 0.4:
        return 0  # Low Risk
    elif risk_score < 0.7:
        return 1  # Moderate Risk
    else:
        return 2  # High Risk

# Apply function to each row
user_features['label'] = user_features.apply(generate_ordinal_label, axis=1)

# --- Step 4: Save Output ---
output_path = os.path.join(output_dir, "user_features_labeled.csv")
try:
    user_features.to_csv(output_path, index=False)
    print(f"✅ Ordinal labels generated and saved at {output_path}")
except Exception as e:
    print(f"❌ Failed to save labeled dataset: {e}")
