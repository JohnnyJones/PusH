import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np


# Create the environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env = TimeLimit(env, max_episode_steps=100)

# Initialize the Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])
print(Q)

# Set learning parameters
y = 0.9
lr = 0.05
num_episodes = 2000

# Record the rewards and steps
j_list = []
r_list = []

# Train the agent
for i in range(num_episodes):
    s, _ = env.reset()
    r_total = 0
    j = 0

    # Episode loop
    while True:
        j += 1
        
        # Pick action at random
        a = np.random.randint(4)

        # Take action, get new state
        s1, r, terminated, truncated, _ = env.step(a)
        r_total += r

        # Possible states to move to and corresponding actions
        Q_max = np.max([Q[s-1, 0], Q[s+4, 1], Q[s+1, 2], Q[s-4, 3]]) # left, down, right, up
        Q[s, a] = Q[s, a] + lr * (r + y * Q_max - Q[s, a])
        s = s1

        if terminated or truncated:
            break


    j_list.append(j)
    r_list.append(r_total)

print("Q-Table Values")
print(Q)