import pandas as pd
import numpy as np
import os
import sys

print(f"✅ Python executing this script: {sys.executable}")
print("Starting tensor decomposition process...")

# --- Dynamic Path Handling ---
script_dir = os.path.dirname(os.path.abspath(__file__))

input_data_path = os.path.join(script_dir, "..", "outputs", "ssl_results", "user_features_ssl_labeled.csv")
absolute_input_path = os.path.abspath(input_data_path)

print(f"DEBUG: script_dir is: {script_dir}")
print(f"DEBUG: Attempting to access file at absolute path: {absolute_input_path}")

# --- File Existence Check ---
if not os.path.exists(absolute_input_path):
    print(f"❌ Error: File not found at {absolute_input_path}")
    sys.exit(1)
try:
    with open(absolute_input_path, 'r') as f:
        print("✅ File exists and is readable.")
except Exception as e:
    print(f"❌ Failed to read file: {e}")
    sys.exit(1)

# --- Output Paths ---
output_factors_dir = os.path.join(script_dir, "..", "models", "tensor_factors")
os.makedirs(output_factors_dir, exist_ok=True)
output_factor_matrices_path = os.path.join(output_factors_dir, "tensor_factors.pkl")

# --- Load Data ---
try:
    df_features = pd.read_csv(input_data_path)
    print(f"✅ Loaded data: {df_features.shape}")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

# --- Tensor Construction ---
if 'user_id' in df_features.columns and 'hour' in df_features.columns:
    tensor_features = df_features.drop(columns=['user_id', 'hour'], errors='ignore').select_dtypes(include=np.number).columns.tolist()

    if not tensor_features:
        print("❌ No numeric features found.")
        sys.exit(1)

    tensor_df = df_features.pivot_table(index='user_id', columns='hour', values=tensor_features, fill_value=0)
    tensor_df.columns = [f'{col[0]}_{col[1]}' for col in tensor_df.columns]
    X_reshaped = tensor_df.values

    n_users = len(tensor_df.index)
    n_hours = len(df_features['hour'].unique())
    n_features_per_hour = len(tensor_features)

    if X_reshaped.shape[1] == n_hours * n_features_per_hour:
        X = X_reshaped.reshape(n_users, n_hours, n_features_per_hour)
        print(f"✅ 3D Tensor shape: {X.shape}")
    else:
        print("❌ Reshape to 3D failed. Falling back to 2D.")
        X = df_features.drop(columns=['user_id'], errors='ignore').select_dtypes(include=np.number).values
else:
    print("⚠️ Falling back to 2D matrix (User x Feature)")
    X = df_features.drop(columns=['user_id'], errors='ignore').select_dtypes(include=np.number).values
    print(f"✅ 2D Matrix shape: {X.shape}")

# --- Tensor Decomposition ---
try:
    print("\n📦 Attempting to import 'tensorly'...")
    import tensorly as tl
    from tensorly.decomposition import tucker, parafac  # 'cp' is now 'parafac'

    if X.ndim == 3:
        tucker_rank = (min(X.shape[0], 10), min(X.shape[1], 5), min(X.shape[2], 10))
        cp_rank = min(X.shape[0], X.shape[1], X.shape[2], 20)

        print(f"🔍 3D Tucker Decomposition (rank={tucker_rank})...")
        core_tucker, factors_tucker = tucker(X, rank=tucker_rank)
        print("✅ Tucker decomposition complete.")

        print(f"🔍 3D CP Decomposition (rank={cp_rank})...")
        factors_cp = parafac(X, rank=cp_rank)
        print("✅ CP decomposition complete.")

    elif X.ndim == 2:
        tucker_rank = (min(X.shape[0], 100), min(X.shape[1], 50))
        cp_rank = min(X.shape[0], X.shape[1], 100)

        print(f"🔍 2D Tucker Decomposition (rank={tucker_rank})...")
        core_tucker, factors_tucker = tucker(X, rank=tucker_rank)
        print("✅ Tucker decomposition complete.")

        print(f"🔍 2D CP Decomposition (rank={cp_rank})...")
        factors_cp = parafac(X, rank=cp_rank)
        print("✅ CP decomposition complete.")

    else:
        print("❌ Unsupported tensor dimensionality.")
        sys.exit(1)

    import joblib
    joblib.dump({
        "tucker_core": core_tucker,
        "tucker_factors": factors_tucker,
        "cp_factors": factors_cp
    }, output_factor_matrices_path)
    print(f"💾 Saved tensor factors to: {output_factor_matrices_path}")

except ImportError:
    print("❌ Tensorly is not installed. Run: pip install tensorly")
    sys.exit(1)
except Exception as e:
    print(f"❌ Tensor decomposition failed: {e}")
    sys.exit(1)

print("\n✅ Tensor modeling process complete.")
