import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac
import os
import pickle
import pandas as pd # Added for potential data loading example

def generate_sample_tensor(users=100, features=5, time_bins=7):
    """
    Generate a random tensor simulating (Users × Features × Time).
    This is a PLACEHOLDER FUNCTION.

    Replace this function with your actual tensor loading/construction logic
    from your social media data.

    Example of how to construct a tensor from your cleaned social media data:
    1. Load 'snmdd_dataset_cleaned.csv' (or user_features_engineered.csv if aggregating features)
       data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "snmdd_dataset_cleaned.csv")
       df_cleaned = pd.read_csv(data_path)
       df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'])

    2. Define your features to include in the tensor (e.g., 'sentiment', 'engagement', 'emotional_word_count')
       # These should be features available per post, which can then be aggregated over time.
       tensor_features = ['sentiment', 'engagement', 'emotional_word_count', 'likes', 'comments', 'is_night']

    3. Define your time granularity (e.g., weekly, daily, monthly).
       Example for weekly aggregation:
       df_cleaned['week_of_year'] = df_cleaned['timestamp'].dt.isocalendar().week.astype(int)
       df_cleaned['year'] = df_cleaned['timestamp'].dt.year
       df_cleaned['time_period'] = df_cleaned['year'].astype(str) + '-' + df_cleaned['week_of_year'].astype(str).str.zfill(2)

    4. Aggregate features for each user for each time bin (e.g., mean sentiment per user per week)
       # Create a multi-index dataframe with all user-time_period combinations to ensure completeness
       all_users = df_cleaned['user_id'].unique()
       all_time_periods = sorted(df_cleaned['time_period'].unique())
       multi_index = pd.MultiIndex.from_product([all_users, all_time_periods], names=['user_id', 'time_period'])

       # Group and aggregate your features
       aggregated_data = df_cleaned.groupby(['user_id', 'time_period'])[tensor_features].mean().reindex(multi_index).fillna(0) # Fill NaN for missing periods
       
    5. Reshape into a 3D NumPy array (Users × Features × TimeBins)
       N_USERS = len(all_users)
       N_FEATURES = len(tensor_features)
       N_TIME_BINS = len(all_time_periods)

       data_tensor = np.zeros((N_USERS, N_FEATURES, N_TIME_BINS))

       user_to_idx = {user: i for i, user in enumerate(all_users)}
       time_to_idx = {time_period: k for k, time_period in enumerate(all_time_periods)}

       for user_id, user_df in aggregated_data.groupby(level='user_id'):
           u_idx = user_to_idx[user_id]
           for time_period, values in user_df.iterrows():
               t_idx = time_to_idx[time_period[1]] # time_period is a tuple (user_id, time_period_str)
               for f_idx, feature_name in enumerate(tensor_features):
                   data_tensor[u_idx, f_idx, t_idx] = values[feature_name]
       return data_tensor
    """
    print("⚠️ Using random sample tensor. Replace 'generate_sample_tensor' with actual data loading logic!")
    return np.random.rand(users, features, time_bins)

def run_tucker_decomposition(tensor, rank=(3, 3, 3), save_path=None):
    """
    Perform Tucker decomposition on the tensor.
    `rank` is a tuple specifying multilinear rank for each mode.
    """
    print(f"\n🚀 Running Tucker Decomposition with rank {rank}...")
    core, factors = tucker(tensor, rank=rank)
    if save_path:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump((core, factors), f)
        print(f"✅ Tucker decomposition saved to {save_path}")
    print("✅ Tucker decomposition complete.")
    return core, factors

def run_cp_decomposition(tensor, rank=3, save_path=None):
    """
    Perform CP decomposition on the tensor.
    `rank` is the number of rank-1 components.
    """
    print(f"\n🚀 Running CP Decomposition with rank {rank}...")
    cp_factors = parafac(tensor, rank=rank)
    if save_path:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(cp_factors, f)
        print(f"✅ CP decomposition saved to {save_path}")
    print("✅ CP decomposition complete.")
    return cp_factors

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Output path for tensor models (SNMD-main/models/tensor_factors/)
    output_dir = os.path.join(script_dir, "models", "tensor_factors")
    os.makedirs(output_dir, exist_ok=True)

    # --- REPLACE THIS WITH YOUR ACTUAL TENSOR ---
    tensor = generate_sample_tensor(users=100, features=6, time_bins=10) # Example dimensions

    # Run Tucker Decomposition
    tucker_core, tucker_factors = run_tucker_decomposition(
        tensor,
        rank=(5, 3, 4), # Example ranks, adjust based on your data and desired compression
        save_path=os.path.join(output_dir, "tucker.pkl")
    )
    print("\n--- Tucker Decomposition Results ---")
    print(f"Core tensor shape: {tucker_core.shape}")
    for i, factor in enumerate(tucker_factors):
        print(f"Factor matrix {i+1} (Mode {i+1}) shape: {factor.shape}")

    # Run CP Decomposition
    cp_factors = run_cp_decomposition(
        tensor,
        rank=5, # Example rank
        save_path=os.path.join(output_dir, "cp.pkl")
    )
    print("\n--- CP Decomposition Results ---")
    for i, factor in enumerate(cp_factors.factors):
        print(f"Factor matrix {i+1} (Mode {i+1}) shape: {factor.shape}")
    print(f"CP Weights: {cp_factors.weights.shape}")

    print("\nTensor decomposition process complete.")