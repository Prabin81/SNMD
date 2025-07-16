import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac
import os
import pickle
import pandas as pd # Added for potential data loading

def generate_sample_tensor(users=100, features=5, time_bins=7):
    
    print("⚠️ Using random sample tensor. Replace 'generate_sample_tensor' with actual data loading logic!")
    return np.random.rand(users, features, time_bins)

def run_tucker_decomposition(tensor, rank=(3, 3, 3), save_path=None):
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