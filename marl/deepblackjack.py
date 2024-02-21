from collections import deque, namedtuple
import gymnasium as gym
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

Replay = namedtuple("Replay", ["state", "action", "reward", "next_state", "terminal"])

class ReplayBuffer(Dataset):
    def __init__(self):
        self.replay = deque()

    def __len__(self):
        return len(self.replay)
    
    def __getitem__(self, index):
        return self.replay[index]
        
    def append(self, item):
        self.replay.append(item)

    def pop(self):
        return self.replay.pop()

def train(env_args, device=torch.device('cpu'), alpha=0.99, y=0.55, lr=0.36, train_episodes=2000):
    # Create the environment
    env = gym.make('Blackjack-v1', **env_args)
    env.reset()

    # Initialize dataset and dataloader
    replay_dataset = ReplayBuffer()
    replay_dataset.append(("empty")) # add dummy item to prevent complaints from dataloader
    replay_dataloader = DataLoader(replay_dataset, batch_size=32, shuffle=True)
    replay_dataset.pop()             # remove dummy item

    # Initialize the DQN
    Q: nn.Module = BlackjackModel()
    Q.to(torch.device(device))
    Q_hat = type(Q)()
    Q_hat.load_state_dict(Q.state_dict())
    Q_hat.to(torch.device(device))

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
    for i in tqdm(range(train_episodes), "Training"):
        s, _ = env.reset()
        s = torch.tensor(s, dtype=torch.float32, device=torch.device(device)).detach()
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
                a = Q(s).argmax().item()

            # Take action, get new state
            s1, r, terminated, truncated, _ = env.step(a)

            # Store the transition as tensors
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=torch.int64, device=torch.device(device)).detach()
            r = torch.tensor(r, dtype=torch.float32, device=torch.device(device)).detach()
            s1 = torch.tensor(s1, dtype=torch.float32, device=torch.device(device)).detach()
            t = torch.tensor(terminated or truncated, dtype=torch.bool, device=torch.device(device)).detach()
            replay_dataset.append(Replay(s, a, r, s1, t))

            # Decay epsilon
            e = max(0.1, e - decay)

            # Update state
            s = s1.clone().detach()

            # Skip training if not enough transitions
            if len(replay_dataset) < 32:
                if terminated or truncated:
                    break
                # else skip training
                continue
            
            # --------------------------------------
            # Update nn
            # --------------------------------------

            # Sample a batch
            batch: Replay = next(iter(replay_dataloader))
            target: torch.Tensor

            with torch.no_grad():
                # Target output calculation
                output_hat: torch.Tensor = Q_hat(batch.state)
                values_hat = torch.max(output_hat, dim=1).values
                target = batch.reward + y * values_hat * ~batch.terminal        

            # Actual output
            values: torch.Tensor = Q(batch.state)
            estimate = values.gather(1, batch.action.unsqueeze(1)).squeeze(1)

            # Gradient descent
            optim.zero_grad()
            loss = loss_fn(estimate, target)
            loss.backward()
            optim.step()

            # Update target network if needed
            if k % c == 0:
                Q_hat.load_state_dict(Q.state_dict())
            
            # Check if episode finished
            if terminated or truncated:
                break
    
    env.close()
            
    return Q, j_list, r_list


def eval(Q, env_args, eval_episodes=1000, device='cpu'):
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
            # Choose the best action
            a = Q(torch.tensor(s, dtype=torch.float32, device=torch.device(device))).argmax().item()

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


def viz(Q, env_args, viz_episodes=10, device='cpu'):
    env = gym.make('Blackjack-v1', render_mode="human", **env_args)
    env.reset()

    for i in range(viz_episodes):
        s, _ = env.reset()
        print("Attempt", i, end=" ")

        # Episode loop
        while True:
            # Choose the best action
            a = Q(torch.tensor(s, dtype=torch.float32, device=torch.device(device))).argmax().item()


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
    train_episodes = 20000
    eval_episodes = 1000000
    viz_episodes = 10

    # Check if CUDA is available
    device = 'cpu'
    if torch.cuda.is_available():
        print("Using CUDA")
        device = 'cuda'

    # Train, eval, and visualize
    Q, j_list, r_list = train(env_args, device=device, train_episodes=train_episodes)
    wins, draws, losses, rewards = eval(Q, env_args, eval_episodes, device=device)
    # if args.viz:
    #     viz(Q, env_args)

    # Print results    
    print(f"Overall win rate: {wins/eval_episodes*100:.2f}%")
    print(f"Overall draw rate: {draws/eval_episodes*100:.2f}%")
    print(f"Overall loss rate: {losses/eval_episodes*100:.2f}%")
    print(f"Average reward: {np.mean(rewards)}")