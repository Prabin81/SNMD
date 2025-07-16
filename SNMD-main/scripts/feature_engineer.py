import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re

# Download required NLTK data (run once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('corpora/words.zip')
except nltk.downloader.DownloadError:
    nltk.download('words')

print("Starting feature engineering process...")

# --- Dynamic Path Handling ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input path: cleaned data from main.py (Now correctly pointing to top-level outputs)
input_path = os.path.join(script_dir, "..", "..", "outputs", "snmdd_dataset_cleaned.csv")

# Output path for engineered features (UNLABELED) (Now correctly pointing to top-level outputs)
output_dir_parent = os.path.join(script_dir, "..", "..", "outputs")
os.makedirs(output_dir_parent, exist_ok=True) # Ensure main outputs directory exists
output_path = os.path.join(output_dir_parent, "user_features_engineered.csv")


# Load the cleaned dataset
try:
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Successfully loaded cleaned data from: {input_path}")
except FileNotFoundError:
    print(f"❌ Error: Cleaned dataset not found at {input_path}. Please run main.py first.")
    exit()

# Ensure engagement column exists (should be created by main.py, but good to check)
if 'engagement' not in df.columns:
    df['engagement'] = df['likes'] + df['comments']

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
# Ensure 'post_text' is treated as string and handle NaN gracefully
df['post_text'] = df['post_text'].astype(str).fillna('')
df = df[df['post_text'].str.strip() != ''] # Remove empty strings after fillna
df['sentiment'] = df['post_text'].apply(lambda text: sia.polarity_scores(text)['compound'])
print("✅ Sentiment analysis complete.")

# Aggregate behavioral features
user_features = df.groupby('user_id').agg(
    avg_likes=('likes', 'mean'),
    avg_comments=('comments', 'mean'),
    avg_engagement=('engagement', 'mean'),
    avg_sentiment=('sentiment', 'mean'),
    neg_post_ratio=('sentiment', lambda x: (x < 0).mean()), # Ratio of negative sentiment posts
    total_posts=('timestamp', 'count')
)
user_features.reset_index(inplace=True)
print("✅ Aggregated basic user features.")

# Posting burstiness / Avg Post Interval
df = df.sort_values(by=['user_id', 'timestamp'])
df['time_gap'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
# Fill NaN in time_gap (first post for each user) with 0 for aggregation, assuming 0 implies very short gap or irrelevant
user_features = pd.merge(user_features,
                         df.groupby('user_id')['time_gap'].mean().reset_index().rename(columns={'time_gap': 'avg_post_interval'}),
                         on='user_id', how='left')
user_features['avg_post_interval'].fillna(0, inplace=True) # Fill for users with single post (diff is NaN)
print("✅ Calculated average post interval.")


# Emotional word usage
emotional_words = ['happy', 'sad', 'angry', 'excited', 'fear', 'love', 'depressed', 'joy', 'anxious', 'stress', 'lonely', 'suicide', 'anxiety', 'panic', 'worthless'] # Expanded list slightly
def count_emotional_words(text):
    tokens = str(text).lower().split()
    return sum(word in emotional_words for word in tokens)

df['emotional_word_count'] = df['post_text_cleaned'].apply(count_emotional_words) # Use cleaned text
emotional_features = df.groupby('user_id')['emotional_word_count'].mean().reset_index()
emotional_features = emotional_features.rename(columns={'emotional_word_count': 'avg_emotional_words'})
user_features = pd.merge(user_features, emotional_features, on='user_id', how='left')
user_features['avg_emotional_words'].fillna(0, inplace=True) # Fill for users with no emotional words
print("✅ Calculated average emotional words.")


# NEW Features

# 1. Night activity ratio (10 PM to 5 AM)
df['is_night'] = df['timestamp'].dt.hour.apply(lambda h: 1 if h >= 22 or h < 5 else 0)
night_ratio = df.groupby('user_id')['is_night'].mean().reset_index()
night_ratio = night_ratio.rename(columns={'is_night': 'night_activity_ratio'})
user_features = pd.merge(user_features, night_ratio, on='user_id', how='left')
user_features['night_activity_ratio'].fillna(0, inplace=True)
print("✅ Calculated night activity ratio.")

# 2. Engagement volatility (standard deviation of engagement)
engagement_std = df.groupby('user_id')['engagement'].std().reset_index()
engagement_std = engagement_std.rename(columns={'engagement': 'engagement_volatility'})
user_features = pd.merge(user_features, engagement_std, on='user_id', how='left')
user_features['engagement_volatility'].fillna(0, inplace=True) # Fill for users with single post (std dev is NaN)
print("✅ Calculated engagement volatility.")

# 3. Social reciprocity (if 'replies' column exists)
if 'replies' in df.columns:
    df['replies'] = pd.to_numeric(df['replies'], errors='coerce').fillna(0)
    # To avoid division by zero, replace 0 replies with 1 in denominator for ratio, or use 0 if replies is 0
    df['reciprocity'] = df.apply(lambda row: row['comments'] / (row['replies'] + 1) if row['replies'] >= 0 else 0, axis=1)
    reciprocity_agg = df.groupby('user_id')['reciprocity'].mean().reset_index()
    user_features = pd.merge(user_features, reciprocity_agg, on='user_id', how='left')
    user_features['reciprocity'].fillna(0, inplace=True) # Fill for users with no replies data
    print("✅ Calculated social reciprocity.")
else:
    print("ℹ️ 'replies' column not found. Skipping social reciprocity feature.")


# --- Feature Validation (NO LABEL GENERATION HERE) ---
def validate_features(df_to_validate):
    """Check for feature validity"""
    assert df_to_validate['avg_sentiment'].between(-1, 1).all(), "Sentiment scores out of range"
    assert df_to_validate['avg_engagement'].ge(0).all(), "Negative engagement values"
    assert df_to_validate['night_activity_ratio'].between(0, 1).all(), "Invalid night activity ratio"
    assert df_to_validate['total_posts'].ge(1).all(), "Zero total posts for a user"
    assert df_to_validate['avg_post_interval'].ge(0).all(), "Negative post interval"

    # Fill NaN with 0 before validation for std, which can be NaN for single posts
    assert df_to_validate['engagement_volatility'].fillna(0).ge(0).all(), "Negative engagement volatility"
    assert df_to_validate['avg_emotional_words'].ge(0).all(), "Negative emotional words count"
    # Add more validations as needed for new features
    print("✅ Features validated successfully!")
    return True

# Save the engineered features (UNLABELED)
try:
    if validate_features(user_features):
        user_features.to_csv(output_path, index=False)
        print(f"✅ Engineered features saved successfully to '{output_path}'")
except AssertionError as e:
    print(f"❌ Feature validation failed: {e}")
    # Still save, but with a warning, or decide to exit based on severity
    user_features.to_csv(output_path, index=False)
    print(f"⚠️ Features saved despite validation warning due to: {e}")
except Exception as e:
    print(f"❌ Failed to save engineered dataset: {e}")

print("Feature engineering process complete.")