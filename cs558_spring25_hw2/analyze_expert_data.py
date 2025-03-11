import pickle
import numpy as np
import argparse

def load_expert_data(file_path):
    """Loads expert trajectories from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_expert_data(file_path):
    """Computes mean and std of return over two trajectories."""
    paths = load_expert_data(file_path)

    print(paths[0]["reward"])


    if len(paths) < 2:
        print(f"Error: Need at least two trajectories, but found {len(paths)} in {file_path}")
        return

    # Extract returns (sum of rewards per trajectory)
    returns = [np.sum(path["reward"]) for path in paths[:2]]

    # Compute statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    print(f"Analysis for {file_path}:")
    print(f"  - Mean Return: {mean_return}")
    print(f"  - Std Dev Return: {std_return}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_data", "-ed", type=str, required=True,
                        help="Path to the expert data pickle file")
    args = parser.parse_args()

    analyze_expert_data(args.expert_data)
