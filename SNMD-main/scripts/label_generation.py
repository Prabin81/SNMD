import pandas as pd
import os

print("Starting label generation process...")

# --- Dynamic Path Handling ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input path: engineered features from feature_engineer.py (UNLABELED)
# Now correctly pointing to top-level outputs
input_path = os.path.join(script_dir, "..", "..", "outputs", "user_features_engineered.csv")

# Output directory for labeled features (Now correctly pointing to top-level outputs)
output_dir = os.path.join(script_dir, "..", "..", "outputs")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
try:
    user_features = pd.read_csv(input_path)
    print(f"✅ Successfully loaded engineered features from {input_path}")
except FileNotFoundError:
    print(f"❌ Error: Engineered features file not found at {input_path}")
    print("💡 Please run feature_engineer.py first to generate the input file.")
    exit()

# --- Step 3: Heuristic Label Generation (This is the SOLE place for this logic) ---
def generate_ordinal_label(row):
    """Multi-dimensional risk assessment with ordinal labels."""
    risk_score = 0

    # Ensure features exist before using them, provide default if missing
    # Using .get() for robustness in case a column is missing
    avg_sentiment = row.get('avg_sentiment', 0.0)
    neg_post_ratio = row.get('neg_post_ratio', 0.0)
    night_activity_ratio = row.get('night_activity_ratio', 0.0)
    engagement_volatility = row.get('engagement_volatility', 0.0)
    avg_emotional_words = row.get('avg_emotional_words', 0.0)
    avg_post_interval = row.get('avg_post_interval', 0.0) # Use 0 for interval, large number might skew

    # Sentiment (30% weight)
    if avg_sentiment < -0.5:
        risk_score += 0.3
    elif avg_sentiment < -0.2:
        risk_score += 0.15
    # Add consideration for high neg_post_ratio
    if neg_post_ratio > 0.5: # More than half posts are negative sentiment
        risk_score += 0.1

    # Engagement (25% weight)
    # Consider users with very low engagement or very high volatility
    if row.get('avg_engagement', 0.0) < 5 and row.get('total_posts', 0) > 5: # Low engagement for active users
        risk_score += 0.15
    if engagement_volatility > 15: # High fluctuation in engagement
        risk_score += 0.1

    # Temporal patterns (25% weight)
    if night_activity_ratio > 0.4: # Significant night activity
        risk_score += 0.25
    if avg_post_interval > 0 and avg_post_interval < 1800 and row.get('total_posts', 0) > 10: # Very frequent posting (less than 30 mins avg, and more than 10 posts)
        risk_score += 0.1

    # Emotional content (20% weight)
    if avg_emotional_words > 2: # High average use of emotional words
        risk_score += 0.2
    
    # Example: If reciprocity feature exists and is low, add risk
    if 'reciprocity' in row and row.get('reciprocity', 1.0) < 0.5 and row.get('total_posts', 0) > 5:
        risk_score += 0.05 # Small added risk for low social reciprocity

    # Label assignment based on updated risk_score thresholds
    # Adjust thresholds based on desired class distribution
    if risk_score < 0.3:
        return 0  # Low Risk
    elif risk_score < 0.6:
        return 1  # Moderate Risk
    else:
        return 2  # High Risk

print("\nApplying heuristic to generate ordinal labels...")
user_features['label'] = user_features.apply(generate_ordinal_label, axis=1)
print("✅ Labels generated.")
print(f"Generated Label Distribution:\n{user_features['label'].value_counts()}")

# --- Step 4: Save Output ---
output_path = os.path.join(output_dir, "user_features_labeled.csv")
try:
    user_features.to_csv(output_path, index=False)
    print(f"✅ Ordinal labels generated and saved at {output_path}")
except Exception as e:
    print(f"❌ Failed to save labeled dataset: {e}")

print("Label generation process complete.")