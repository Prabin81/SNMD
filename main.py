import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("snmdd_dataset.csv")  # Ensure the file name matches exactly
print(df.head())  # Display first 5 rows of the dataset

# Show column info and missing values
print(df.info())
print(df.isnull().sum())

# Drop or fill missing values
df = df.dropna()  # Or use df.fillna(method='ffill') to fill missing data

# Convert timestamp to datetime and extract date/hour features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# Add engagement feature (likes + comments)
df['engagement'] = df['likes'] + df['comments']

# 1. Plot activity over time
daily_posts = df.groupby('date').size()
daily_posts.plot(figsize=(10, 4), title='Posts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.tight_layout()
plt.show()

# 2. Plot likes distribution
sns.histplot(df['likes'], bins=30, kde=True)
plt.title("Likes Distribution")
plt.show()

# 3. Plot comments distribution
sns.histplot(df['comments'], bins=30, kde=True)
plt.title("Comments Distribution")
plt.show()

# 4. Engagement heatmap (hourly activity by date)
heatmap_data = df.groupby(['hour', 'date']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu')
plt.title("Post Frequency Heatmap by Hour and Date")
plt.tight_layout()
plt.show()

# Save cleaned dataset to a new file
df.to_csv("snmdd_dataset_cleaned.csv", index=False)
print("✅ Cleaned data saved successfully to 'snmdd_dataset_cleaned.csv'")
