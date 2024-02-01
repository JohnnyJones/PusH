import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np


# Create the environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env = TimeLimit(env, max_episode_steps=100)
env.reset()

# Initialize the Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])
print("Q-table before:")
print(Q)

# Set learning parameters
y = 0.9
lr = 0.5
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

        # Choose the best action if it exists
        if np.max(Q[s]) > 0:
            a = np.argmax(Q[s])
        # Or pick action at random
        else:
            a = env.action_space.sample()

        # Take action, get new state
        s1, r, terminated, truncated, _ = env.step(a)
        r_total += r

        # Update Q-table
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1]) - Q[s, a])
        
        # Update state
        s = s1

        if terminated or truncated:
            break

    j_list.append(j)
    r_list.append(r_total)


print("Q-Table after:")
print(Q)

# Evaluation
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
env = TimeLimit(env, max_episode_steps=100)
env.reset()
eval_episodes = 10

for i in range(eval_episodes):
    s, _ = env.reset()
    print("Attempt", i, end="")

    # Episode loop
    while True:
        j += 1

        # Choose the best action if it exists
        if np.max(Q[s]) > 0:
            a = np.argmax(Q[s])
        # Or pick action at random
        else:
            a = env.action_space.sample()

        # Take action, get new state
        s1, r, terminated, truncated, _ = env.step(a)
        r_total += r

        # Penalty for falling in hole
        if terminated and r <= 0:
            r = -1
        
        # Update state
        s = s1

        if terminated or truncated:
            break
    
    if r == 1.0:
        print(" Success")
    else:
        print(" Failure")