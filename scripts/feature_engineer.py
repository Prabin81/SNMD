import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('words')

# === Step 1: Load the cleaned dataset ===
df = pd.read_csv("F:/SNMD/SNMD/snmdd_dataset_cleaned.csv")  # Update path if needed
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Ensure engagement column exists
if 'engagement' not in df.columns:
    df['engagement'] = df['likes'] + df['comments']

# === Step 2: Sentiment Analysis ===
sia = SentimentIntensityAnalyzer()
df = df[df['post_text'].notna() & (df['post_text'].str.strip() != '')]
df['sentiment'] = df['post_text'].apply(lambda text: sia.polarity_scores(str(text))['compound'])

# === Step 3: Aggregate behavioral features ===
user_features = df.groupby('user_id').agg({
    'likes': 'mean',
    'comments': 'mean',
    'engagement': 'mean',
    'sentiment': ['mean', lambda x: (x < 0).mean()],
    'timestamp': 'count'
})
user_features.columns = [
    'avg_likes',
    'avg_comments',
    'avg_engagement',
    'avg_sentiment',
    'neg_post_ratio',
    'total_posts'
]
user_features.reset_index(inplace=True)

# Posting burstiness
df = df.sort_values(by=['user_id', 'timestamp'])
df['time_gap'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
burstiness = df.groupby('user_id')['time_gap'].mean().reset_index()
burstiness = burstiness.rename(columns={'time_gap': 'avg_post_interval'})
user_features = pd.merge(user_features, burstiness, on='user_id', how='left')

# === Step 4: Emotional word usage ===
emotional_words = ['happy', 'sad', 'angry', 'excited', 'fear', 'love', 'depressed', 'joy', 'anxious']
def count_emotional_words(text):
    tokens = str(text).lower().split()
    return sum(word in emotional_words for word in tokens)

df['emotional_word_count'] = df['post_text'].apply(count_emotional_words)
emotional_features = df.groupby('user_id')['emotional_word_count'].mean().reset_index()
emotional_features = emotional_features.rename(columns={'emotional_word_count': 'avg_emotional_words'})
user_features = pd.merge(user_features, emotional_features, on='user_id', how='left')

# === Step 5: NEW Features (Before Labeling) ===

# 1. Night activity ratio (10 PM to 5 AM)
df['is_night'] = df['timestamp'].dt.hour.apply(lambda h: 1 if h >= 22 or h < 5 else 0)
night_ratio = df.groupby('user_id')['is_night'].mean().reset_index()
night_ratio = night_ratio.rename(columns={'is_night': 'night_activity_ratio'})
user_features = pd.merge(user_features, night_ratio, on='user_id', how='left')

# 2. Engagement volatility
engagement_std = df.groupby('user_id')['engagement'].std().reset_index()
engagement_std = engagement_std.rename(columns={'engagement': 'engagement_volatility'})
user_features = pd.merge(user_features, engagement_std, on='user_id', how='left')

# 3. Social reciprocity (if 'replies' column exists)
if 'replies' in df.columns:
    df['replies'] = pd.to_numeric(df['replies'], errors='coerce').fillna(0)
    df['reciprocity'] = df['comments'] / (df['replies'] + 1)
    reciprocity = df.groupby('user_id')['reciprocity'].mean().reset_index()
    user_features = pd.merge(user_features, reciprocity, on='user_id', how='left')

# === Step 6: Ordinal Labeling ===
def generate_ordinal_label(row):
    score = row['avg_sentiment'] + row['neg_post_ratio'] + row['night_activity_ratio']
    if score < 0.5:
        return 0  # Low Risk
    elif score < 1.5:
        return 1  # Moderate Risk
    else:
        return 2  # High Risk

user_features['label'] = user_features.apply(generate_ordinal_label, axis=1)

# === Step 7: Validate Features ===
def validate_features(df):
    """Check for feature validity"""
    assert df['avg_sentiment'].between(-1, 1).all(), "Sentiment scores out of range"
    assert df['avg_engagement'].ge(0).all(), "Negative engagement values"
    assert df['night_activity_ratio'].between(0, 1).all(), "Invalid night activity ratio"
    return True

# === Step 8: Save the final dataset ===
try:
    if validate_features(user_features):
        user_features.to_csv("user_features_final.csv", index=False)
        print("✅ Features validated and saved successfully!")
except AssertionError as e:
    print(f"❌ Feature validation failed: {e}")
