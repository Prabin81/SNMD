import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac
import os
import pickle

def generate_sample_tensor(users=100, features=5, time_bins=7):
    """
    Generate a random tensor simulating (Users × Features × Time).
    Replace this function with your actual tensor loading logic.
    """
    return np.random.rand(users, features, time_bins)

def run_tucker_decomposition(tensor, rank=(3, 3, 3), save_path=None):
    """
    Perform Tucker decomposition on the tensor.
    `rank` is a tuple specifying multilinear rank for each mode.
    """
    core, factors = tucker(tensor, rank=rank)
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump((core, factors), f)
        print(f"✅ Tucker decomposition saved to {save_path}")
    return core, factors

def run_cp_decomposition(tensor, rank=3, save_path=None):
    """
    Perform CP decomposition on the tensor.
    `rank` is the number of rank-1 components.
    """
    cp_factors = parafac(tensor, rank=rank)
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(cp_factors, f)
        print(f"✅ CP decomposition saved to {save_path}")
    return cp_factors

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "models", "tensor_factors")
    os.makedirs(output_dir, exist_ok=True)

    # Generate or load your tensor here
    tensor = generate_sample_tensor()

    # Run Tucker Decomposition
    run_tucker_decomposition(
        tensor,
        rank=(3, 3, 3),
        save_path=os.path.join(output_dir, "tucker.pkl")
    )

    # Run CP Decomposition
    run_cp_decomposition(
        tensor,
        rank=3,
        save_path=os.path.join(output_dir, "cp.pkl")
    )
