import pandas as pd
import os

# Get absolute path of current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full path to the labeled data file
input_path = os.path.join(script_dir, "output", "user_features_labeled.csv")

# Load data
try:
    data = pd.read_csv(input_path)
    print(f"✅ Loaded labeled data from {input_path}")
except FileNotFoundError:
    print(f"❌ File not found at {input_path}")
    print("💡 Please run label_generation.py first to generate the labeled dataset.")
    exit()
