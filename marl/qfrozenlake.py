import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import argparse


parser = argparse.ArgumentParser(
    prog='QFrozenLake',
    description='Plays FrozenLake using Q-Learning',
)
parser.add_argument('-v', '--viz', action='store_true')
parser.add_argument('-s', '--slippery', action='store_true')
parser.add_argument('-q', '--show-qtable', action='store_true')
args = parser.parse_args()

# Create the environment
env_args = {
    'desc': None,
    'map_name': "4x4",
    'is_slippery': args.slippery,
}
env = gym.make('FrozenLake-v1', **env_args)
env = TimeLimit(env, max_episode_steps=100)
env.reset()

# Initialize the Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])
if args.show_qtable:
    print("Q-table before:")
    print(Q)

# Set learning parameters
y = 0.9
lr = 0.5
e = 1.0
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

        # Explore or choose action with epsilon linear decay
        if np.random.rand() < e - 1.0 / num_episodes:
            a = env.action_space.sample()
        else:
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

if args.show_qtable:
    print("Q-Table after:")
    print(Q)

# Evaluation
env.reset()
eval_episodes = 1000
wins = 0
losses = 0
rewards = []

for i in range(eval_episodes):
    s, _ = env.reset()

    # Episode loop
    while True:
        j += 1

        # Choose the best action
        a = np.argmax(Q[s])

        # Take action, get new state
        s1, r, terminated, truncated, _ = env.step(a)
        r_total += r

        # Update state
        s = s1

        if terminated or truncated:
            break
    
    if r == 1.0:
        wins += 1
    else:
        losses += 1

    rewards.append(r)


# Visualization
if args.viz:
    env = gym.make('FrozenLake-v1', render_mode="human", **env_args)
    env = TimeLimit(env, max_episode_steps=100)
    env.reset()
    viz_episodes = 10

    for i in range(viz_episodes):
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

            # Update state
            s = s1

            if terminated or truncated:
                break
        
        if r == 1.0:
            print(" Success")
        else:
            print(" Failure")

print(f"Overall win rate: {wins/eval_episodes*100:.2f}%")
print(f"Overall loss rate: {losses/eval_episodes*100:.2f}%")
print(f"Average reward: {np.mean(rewards)}")

env.close()