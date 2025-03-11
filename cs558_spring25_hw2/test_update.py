import torch
from cs558.policies.MLP_policy import MLPPolicySL

def test_update():
    print("Testing update() function...")

    policy = MLPPolicySL(ac_dim=4, ob_dim=10, n_layers=2, size=64, discrete=False)

    obs = torch.randn(5, 10)  # Simulate 5 states
    actions = torch.randn(5, 4)  # Simulate 5 expert actions

    loss = policy.update(obs, actions)
    
    print("Update function output:", loss)

if __name__ == "__main__":
    test_update()
