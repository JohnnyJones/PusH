import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import argparse
from tqdm import tqdm

def train(env_args, y=0.55, lr=0.36, train_episodes=2000):
    # Create the environment
    env = gym.make('Blackjack-v1', **env_args)
    env.reset()

    # Initialize the Q-table
    q_size = [env.observation_space[i].n for i in range(len(env.observation_space))] + [env.action_space.n]
    Q = np.zeros(q_size)

    # Record the rewards and steps
    j_list = []
    r_list = []

    # Epsilon
    e = 1.0
    decay = 1.0 / train_episodes

    # Train the agent
    for i in range(train_episodes):
        s, _ = env.reset()
        r_total = 0
        j = 0

        # Episode loop
        while True:
            j += 1

            # Explore or choose action with epsilon
            if np.random.rand() < e:
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
        
        # Decay epsilon
        e -= decay

        j_list.append(j)
        r_list.append(r_total)
    
    env.close()
    
    return Q, j_list, r_list

def eval(Q, env_args, eval_episodes=1000):
    # Create the environment
    env = gym.make('Blackjack-v1', **env_args)
    env.reset()

    # Record the results
    wins = 0
    draws = 0
    losses = 0
    rewards = []

    for i in range(eval_episodes):
        s, _ = env.reset()

        # Episode loop
        while True:
            # Choose the best action
            a = np.argmax(Q[s])

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

def viz(Q, env_args, viz_episodes=10):
    env = gym.make('Blackjack-v1', render_mode="human", **env_args)
    env.reset()

    for i in range(viz_episodes):
        s, _ = env.reset()
        print("Attempt", i, end=" ")

        # Episode loop
        while True:
            # Choose the best action
            a = np.argmax(Q[s])

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
        prog='QBlackjack',
        description='Plays Blackjack using Q-Learning',
    )
    parser.add_argument('-v', '--viz', action='store_true')
    args = parser.parse_args()

    # Environment arguments
    env_args = {
        'natural': False,
        'sab': False,
    }

    # Lengths of training and eval
    train_episodes = 2000
    eval_episodes = 1000
    viz_episodes = 10

    # Train, eval, and visualize
    Q, j_list, r_list = train(env_args, train_episodes=train_episodes)
    wins, draws, losses, rewards = eval(Q, env_args, eval_episodes)
    if args.viz:
        viz(Q, env_args)
    
    print(f"Overall win rate: {wins/eval_episodes*100:.2f}%")
    print(f"Overall draw rate: {draws/eval_episodes*100:.2f}%")
    print(f"Overall loss rate: {losses/eval_episodes*100:.2f}%")
    print(f"Average reward: {np.mean(rewards)}")
