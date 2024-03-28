import argparse
import gymnasium as gym
import random
import time
import numpy as np
import pandas as pd
import torch

from environment import ChineseCheckersEnv
from data import Action, Position
from agent import ChineseCheckersAgent, RandomAgent, DeterministicGreedyAgent, StochasticGreedyAgent, DeepMctsAgent
from tqdm import tqdm
from argparse import ArgumentParser
from collections import deque
from compare import take_random_action
from board import Board
from torch.utils.data import Dataset, DataLoader

class GameDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X, y_truth, y_value = self.data[idx]
        return X, y_truth, y_value

def game_generation(game_count, agents: list[ChineseCheckersAgent] = [DeterministicGreedyAgent(), DeterministicGreedyAgent()], 
                    random_turns=3, random_move_rate=0.5, shuffle_rate=0.3, keep=0.05, as_list=True, replay=True) -> list | GameDataset:
    random_move_games = game_count * random_move_rate
    shuffle_games = random_move_games + game_count * shuffle_rate
    
    env = gym.make("ChineseCheckers-v0")
    max_turns = 100

    # recording game states and ground truth
    states = []
    policy_ground_truths = []
    values = []
    actions = []

    shuffle_start = False

    for game in tqdm(range(game_count), desc="Generating games"):
        if random_move_games < game < shuffle_games:
            shuffle_start = True
        obs, info = env.reset(options={"shuffle_start": shuffle_start})

        game_turn = 0
        game_states = []
        game_truths = []
        game_turns = []
        terminated = False
        while not terminated:
            game_states.append(info["board"].to_tensor())
            best_actions = _get_best_actions(info["board"])
            truth = torch.zeros(size=[6, 7, 7])
            for action in best_actions:
                truth[action.piece_id, action.position.x, action.position.y] = 1 / len(best_actions)
            game_truths.append(truth)
            game_turns.append(info["turn"])

            if not shuffle_start and game < random_move_games and game_turn < len(agents) * random_turns:
                action = take_random_action(obs, info)
            else:
                agent_turn = info["turn"]
                action = agents[agent_turn].act(obs, info)
            # TODO: check for circular actions
            if replay:
                actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            game_turn += 1

            if game_turn > max_turns:
                break

        if reward != 0:
            # game did not tie, we will use this training data
            winner = info["winner"]
            states.extend(game_states)
            policy_ground_truths.extend(game_truths)
            values.extend([-1 if winner == turn else 1 for turn in game_turns])
    
    data = list(zip(states, policy_ground_truths, values))
    data = random.choices(data, k=int(keep*len(data)))
    if replay:
        if as_list:
            return data, replay
        else:
            return GameDataset(data), replay
    else:
        if as_list:
            return data
        else:
            return GameDataset(data)
            
def self_play(game_count, agents: list[DeepMctsAgent], random_turns=6, keep=0.5, temp_turns=10, temperature=2) -> tuple[GameDataset, DeepMctsAgent, float]:
    env = gym.make("ChineseCheckers-v0")
    max_turns = 100

    # recording game states and ground truth
    states = []
    policy_ground_truths = []
    values = []
    winners = [0] * len(agents)
    
    # pre-flip so that the first play unflips
    winners = winners[::-1]
    agents = agents[::-1]

    # set agent temperature
    for i in range(len(agents)):
        agents[i].temperature = temperature

    for game in tqdm(range(game_count), desc="Generating games"):              
        obs, info = env.reset()

        winners = winners[::-1]
        agents = agents[::-1]

        game_turn = 0
        game_states = []
        game_truths = []
        game_turns = []
        terminated = False
        while not terminated:
            game_states.append(info["board"].to_tensor())
            best_actions = _get_best_actions(info["board"])
            truth = torch.zeros(size=[6, 7, 7])
            for action in best_actions:
                truth[action.piece_id, action.position.x, action.position.y] = 1 / len(best_actions)
            game_truths.append(truth)
            game_turns.append(info["turn"])

            if game_turn == (len(agents) * (random_turns + temp_turns)):
                for i in range(len(agents)):
                    agents[i].set_temperature_low()

            if game_turn < len(agents) * random_turns:
                action = take_random_action(obs, info)
            else:
                agent_turn = info["turn"]
                action = agents[agent_turn].act(obs, info)
            # TODO: check for circular actions
            obs, reward, terminated, truncated, info = env.step(action)
            game_turn += 1

            if game_turn > max_turns:
                break

        if reward != 0:
            # game did not tie, we will use this training data
            winner = info["winner"]
            winners[winner] += 1
            states.extend(game_states)
            policy_ground_truths.extend(game_truths)
            values.extend([-1 if winner == turn else 1 for turn in game_turns])
    
    data = list(zip(states, policy_ground_truths, values))
    data = random.choices(data, k=int(keep*len(data)))

    win_rates = [wins / sum(winners) for wins in winners]
    winner_idx = np.argmax(win_rates)
    best_win_rate = win_rates[winner_idx]
    best_agent = agents[winner_idx]

    return GameDataset(data), best_agent, best_win_rate



def _get_best_actions(board: Board) -> list[Action]:
    agent = DeterministicGreedyAgent()
    actions = board.get_valid_actions_list()
    turn = board.turn

    df = pd.DataFrame({"action": actions})
    df["start"] = [board.get_position_by_id(turn, action.piece_id) for action in actions]
    df["end"] = [action.position for action in actions]
    df["heuristic"] = [agent._heuristic(row["start"], row["end"], turn) for _, row in df.iterrows()]

    max_heuristic = df["heuristic"].max()
    df = df[df["heuristic"] == max_heuristic]

    return df["action"].to_list()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-g", "--games", type=int, default=1000)
    args = parser.parse_args()

    data = game_generation(args.games, [DeterministicGreedyAgent(), DeterministicGreedyAgent()])

    torch.save(data, "./datasets/greedy_play_data")