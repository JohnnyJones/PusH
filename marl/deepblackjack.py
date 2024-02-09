import gymnasium as gym
import numpy as np
import argparse
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import qblackjack as qbj

class BlackjackModel(nn.Module):
    def __init__(self):
        super(BlackjackModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(env_args, alpha=0.99, y=0.55, lr=0.36, train_episodes=2000):
    # Create the environment
    env = gym.make('Blackjack-v1', **env_args)
    env.reset()

    # Initialize the DQN
    replay = []
    Q: nn.Module = BlackjackModel()
    Q_hat = type(Q)()
    Q_hat.load_state_dict(Q.state_dict())

    # Optimizer
    optim = torch.optim.RMSprop(Q.parameters(), lr=lr, alpha=alpha, eps=0.01)

    # Loss
    loss_fn = torch.nn.MSELoss()

    # Record the rewards and steps
    j_list = []
    r_list = []

    # Epsilon
    e = 1.0
    decay = 1.0 / train_episodes

    # How often to update the target network
    c = 100
    k = 0

    # Train the agent
    for i in tqdm(range(train_episodes)):
        s, _ = env.reset()
        r_total = 0
        j = 0

        # Episode loop
        while True:
            j += 1
            k += 1

            # Explore or choose action with epsilon
            if np.random.rand() < e:
                a = env.action_space.sample()
            else:
                # Choose the best action
                a = Q(torch.tensor(s, dtype=torch.float32)).argmax().item()
            
            # Take action, get new state
            s1, r, terminated, truncated, _ = env.step(a)

            # Decay epsilon
            e = max(0.1, e - decay)

            # Store the transition
            replay.append((s, a, r, s1, terminated))

            # Skip training if not enough transitions
            if len(replay) < 32:
                continue

            # Sample a batch
            batch = random.sample(replay, 32)
            target = []
            for state, action, reward, state1, terminal in batch:
                if terminal:
                    target.append(reward)
                else:
                    with torch.no_grad():
                        target.append(reward + y * Q_hat(torch.tensor(state1, dtype=torch.float32)).max())
            target = torch.tensor(target)

            # Forward pass
            estimate = Q(torch.tensor([state for state, _, _, _, _ in batch], dtype=torch.float32)).gather(1, torch.tensor([[action] for _, action, _, _, _ in batch])).squeeze()

            # Gradient descent
            optim.zero_grad()
            loss = loss_fn(estimate, target)
            loss.backward()
            optim.step()

            # Update target network if needed
            if k % c == 0:
                Q_hat.load_state_dict(Q.state_dict())
            
    return Q, j_list, r_list


def eval(Q, env_args, eval_episodes=1000):
    return qbj.eval(Q, env_args, eval_episodes)

def viz(Q, env_args, viz_episodes=10):
    qbj.viz(Q, env_args, viz_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='DeepBlackjack',
        description='Plays Blackjack using Deep Q-Learning',
    )
    parser.add_argument('-v', '--viz', action='store_true')
    args = parser.parse_args()

    # Environment arguments
    env_args = {
        'natural': False,
        'sab': False,
    }

    # Lengths of training and eval
    train_episodes = 200
    eval_episodes = 1000
    viz_episodes = 10

    # Train, eval, and visualize
    Q, j_list, r_list = train(env_args, train_episodes=train_episodes)
    # wins, draws, losses, rewards = eval(Q, env_args, eval_episodes)
    # if args.viz:
    #     viz(Q, env_args)

    # Print results    
    print(f"Overall win rate: {wins/eval_episodes*100:.2f}%")
    print(f"Overall draw rate: {draws/eval_episodes*100:.2f}%")
    print(f"Overall loss rate: {losses/eval_episodes*100:.2f}%")
    print(f"Average reward: {np.mean(rewards)}")