import numpy as np
import torch
from cs558.policies.MLP_policy import MLPPolicy

# Dummy policy parameters
ac_dim = 4  # Example action space dimension
ob_dim = 10  # Example observation space dimension
n_layers = 2
size = 64

# Create a test policy
policy = MLPPolicy(
    ac_dim=ac_dim, ob_dim=ob_dim, n_layers=n_layers, size=size, discrete=False
)

# Generate a random observation (simulating environment input)
test_obs = np.random.randn(ob_dim)  # 1D NumPy array with ob_dim values

# Get action
action = policy.get_action(test_obs)

# Print result
print("Test observation:", test_obs)
print("Generated action:", action)
