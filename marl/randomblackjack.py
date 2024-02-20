import gymnasium as gym
import numpy as np
import argparse

from tqdm import tqdm

def eval(env_args, eval_episodes=1000):
    # Create the environment
    env = gym.make('Blackjack-v1', **env_args)
    env.reset()

    # Record the results
    wins = 0
    draws = 0
    losses = 0
    rewards = []

    for i in tqdm(range(eval_episodes), "Evaluation"):
        s, _ = env.reset()

        # Episode loop
        while True:
            # Choose action at random
            a = env.action_space.sample()

            # Take action, get new state
            s1, r, terminated, truncated, _ = env.step(a)

            # Update state
            s = s1

            if terminated or truncated:
                break
        
        # Record the result
        if r >= 1.0:
            wins += 1
        elif r == -1.0:
            losses += 1
        elif r == 0.0:
            draws += 1
        
        rewards.append(r)
    
    env.close()
    
    return wins, draws, losses, rewards

def viz(env_args, viz_episodes=10):
    env = gym.make('Blackjack-v1', render_mode="human", **env_args)
    env.reset()

    for i in range(viz_episodes):
        s, _ = env.reset()
        print("Attempt", i, end=" ")

        # Episode loop
        while True:
            # Choose random action
            a = env.action_space.sample()

            # Take action, get new state
            s1, r, terminated, truncated, _ = env.step(a)

            # Update state
            s = s1

            if terminated or truncated:
                break
        
        if r >= 1.0:
            print("Win")
        elif r == -1.0:
            print("Loss")
        elif r == 0.0:
            print("Draw")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RandomBlackjack',
        description='Plays Blackjack randomly',
    )
    parser.add_argument('-v', '--viz', help='use a visualizer to show example play', action='store_true')
    args = parser.parse_args()

    # Environment arguments
    env_args = {
        'natural': False,
        'sab': False,
    }

    # Lengths of eval
    eval_episodes = 1000000
    viz_episodes = 10

    # Train, eval, and visualize
    wins, draws, losses, rewards = eval(env_args, eval_episodes)
    if args.viz:
        viz(env_args)
    
    # Print results    
    print(f"Overall win rate: {wins/eval_episodes*100:.2f}%")
    print(f"Overall draw rate: {draws/eval_episodes*100:.2f}%")
    print(f"Overall loss rate: {losses/eval_episodes*100:.2f}%")
    print(f"Average reward: {np.mean(rewards)}")
