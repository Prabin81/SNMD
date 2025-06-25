import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data (run once)
nltk.download('vader_lexicon')
nltk.download('words')

# === Step 1: Load your main cleaned dataset ===
df = pd.read_csv("F:\SNMD\SNMD\snmdd_dataset_cleaned.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# If engagement column missing, create it
if 'engagement' not in df.columns:
    df['engagement'] = df['likes'] + df['comments']

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Filter out empty posts before sentiment analysis
df = df[df['post_text'].notna() & (df['post_text'].str.strip() != '')]

# Calculate sentiment compound score for each post
df['sentiment'] = df['post_text'].apply(lambda text: sia.polarity_scores(str(text))['compound'])

# Aggregate user-level features
user_features = df.groupby('user_id').agg({
    'likes': 'mean',
    'comments': 'mean',
    'engagement': 'mean',
    'sentiment': ['mean', lambda x: (x < 0).mean()],
    'timestamp': 'count'
})

# Rename columns for clarity
user_features.columns = [
    'avg_likes',
    'avg_comments',
    'avg_engagement',
    'avg_sentiment',
    'neg_post_ratio',
    'total_posts'
]
user_features.reset_index(inplace=True)

# Calculate posting burstiness (average time between posts)
df = df.sort_values(by=['user_id', 'timestamp'])
df['time_gap'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
burstiness = df.groupby('user_id')['time_gap'].mean().reset_index()
burstiness = burstiness.rename(columns={'time_gap': 'avg_post_interval'})

# Merge burstiness with user features
user_features = pd.merge(user_features, burstiness, on='user_id', how='left')


# === Step 3 (Optional): Add Self-disclosure / Emotional Cues Feature ===
emotional_words = ['happy', 'sad', 'angry', 'excited', 'fear', 'love', 'depressed', 'joy', 'anxious']

def count_emotional_words(text):
    tokens = str(text).lower().split()
    return sum(word in emotional_words for word in tokens)

df['emotional_word_count'] = df['post_text'].apply(count_emotional_words)

# Aggregate emotional word count per user
emotional_features = df.groupby('user_id').agg({
    'emotional_word_count': 'mean'
}).rename(columns={'emotional_word_count': 'avg_emotional_words'})

# Merge emotional features into user features
user_features = pd.merge(user_features, emotional_features, on='user_id', how='left')

# === Step 4: Generate Labels for Semi-Supervised Learning ===
def generate_labels(row):
    # Customize thresholds as per your project needs
    if row['avg_sentiment'] < -0.2 and row['avg_engagement'] < 10:
        return 1  # At risk / Positive class
    else:
        return 0  # Negative class

user_features['label'] = user_features.apply(generate_labels, axis=1)

# === Step 5: Save the final dataset ===
user_features.to_csv("user_features_final.csv", index=False)
print("✅ user_features_final.csv saved successfully!")
