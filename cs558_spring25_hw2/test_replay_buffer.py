import numpy as np
from cs558.infrastructure.replay_buffer import ReplayBuffer

def test_sample_random_data():
    print("Testing sample_random_data()...")

    # Step 1: Create a ReplayBuffer with max size 100
    buffer = ReplayBuffer(max_size=100)

    # Step 2: Generate some fake data
    num_samples = 50  # Number of samples to store in the buffer
    ob_dim = 4        # Example observation dimension
    ac_dim = 2        # Example action dimension

    fake_paths = []
    num_steps_per_traj = 2  # Number of steps in each trajectory (you can increase this)
    for _ in range(num_samples):
        path = {
            "observation": np.random.randn(num_steps_per_traj, ob_dim),  # Sequence of states
            "action": np.random.randn(num_steps_per_traj, ac_dim),       # Sequence of actions
            "reward": np.random.randn(num_steps_per_traj),               # Sequence of rewards
            "next_observation": np.random.randn(num_steps_per_traj, ob_dim),  # Sequence of next states
            "terminal": np.random.randint(0, 2, size=(num_steps_per_traj,)),   # Sequence of done flags
        }
        fake_paths.append(path)

    # Step 3: Add data to the replay buffer
    buffer.add_rollouts(fake_paths)

    # ✅ Print the buffer size before sampling
    print(f"Replay buffer size after adding rollouts: {len(buffer)}")

    # ✅ Debug: Print stored data
    print(f"Obs shape: {None if buffer.obs is None else buffer.obs.shape}")
    print(f"Acs shape: {None if buffer.acs is None else buffer.acs.shape}")
    print(f"Rews shape: {None if buffer.rews is None else buffer.rews.shape}")
    print(f"Next Obs shape: {None if buffer.next_obs is None else buffer.next_obs.shape}")
    print(f"Terminals shape: {None if buffer.terminals is None else buffer.terminals.shape}")


    # Step 4: Sample random data
    batch_size = 5
    obs, acs, rews, next_obs, terminals = buffer.sample_random_data(batch_size)

    # Step 5: Check results
    print(f"Sampled {batch_size} random states:\n", obs)
    print(f"Sampled {batch_size} random actions:\n", acs)
    print(f"Sampled {batch_size} random rewards:\n", rews)
    print(f"Sampled {batch_size} random next states:\n", next_obs)
    print(f"Sampled {batch_size} random terminals:\n", terminals)

# Run the test
if __name__ == "__main__":
    test_sample_random_data()
