import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

print("Starting main data cleaning and initial EDA process...")

# --- Dynamic Path Handling ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input path: Original raw data (adjust 'output' if your raw data is elsewhere)
# Based on your structure, it looks like original raw 'snmdd_dataset.csv' is under SNMD-main/data/
input_data_path = os.path.join(script_dir, "data", "E:\SNMD-main\SNMD-main\output\snmdd_dataset.csv")

# Output directory for cleaned data (should be the top-level outputs folder)
# From SNMD-main/, go into 'outputs'
output_dir_parent = os.path.join(script_dir, "outputs")
os.makedirs(output_dir_parent, exist_ok=True) # Ensure the main outputs directory exists
output_cleaned_path = os.path.join(output_dir_parent, "snmdd_dataset_cleaned.csv")

# --- Load Data ---
try:
    df = pd.read_csv(input_data_path)
    print(f"✅ Successfully loaded raw data from: {input_data_path}")
    print(f"Initial dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: Raw dataset not found at {input_data_path}.")
    print("Please ensure 'snmdd_dataset.csv' is in the correct 'data' subfolder.")
    exit()
except Exception as e:
    print(f"❌ An error occurred while loading raw data: {e}")
    exit()

# --- Initial EDA and Cleaning ---
print("\n--- Initial Data Overview ---")
print(df.head())
print("\n--- Column Information ---")
df.info()
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Basic Cleaning: Drop rows with essential missing values
initial_rows = df.shape[0]
df.dropna(subset=['user_id', 'post_text', 'timestamp'], inplace=True)
print(f"\n✅ Dropped rows with missing essential values. Rows removed: {initial_rows - df.shape[0]}")

# Convert timestamp to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
# Drop rows where timestamp conversion failed
df.dropna(subset=['timestamp'], inplace=True)
print(f"✅ Converted 'timestamp' to datetime and handled errors. New shape: {df.shape}")

# Handle duplicate posts (based on user_id and post_text)
df.drop_duplicates(subset=['user_id', 'post_text', 'timestamp'], inplace=True)
print(f"✅ Removed duplicate posts. New shape: {df.shape}")

# Ensure 'likes' and 'comments' are numeric, fill NaN with 0
df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0).astype(int)
print("✅ Converted 'likes' and 'comments' to numeric, filled NaNs with 0.")

# Calculate 'engagement'
df['engagement'] = df['likes'] + df['comments']
print("✅ Calculated 'engagement' feature.")

# Extract time-based features
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
print("✅ Extracted 'date' and 'hour' features.")

# Basic text cleaning (e.g., remove URLs, special characters)
def clean_text(text):
    text = str(text) # Ensure text is string
    text = re.sub(r'http\S+', '', text)   # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    text = text.lower() # Convert to lowercase
    return text

df['post_text_cleaned'] = df['post_text'].apply(clean_text)
print("✅ Performed basic text cleaning on 'post_text'.")


# --- Save Cleaned Data ---
try:
    df.to_csv(output_cleaned_path, index=False)
    print(f"\n✅ Cleaned dataset saved successfully to '{output_cleaned_path}'")
except Exception as e:
    print(f"❌ Failed to save cleaned dataset: {e}")

print("\nMain data cleaning and initial EDA process complete.")