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
                    random_turns=3, random_move_rate=0.5, shuffle_rate=0.3, keep=0.05, flip_double=True, as_list=True, replay=True) -> list | GameDataset:
    random_move_games = game_count * random_move_rate
    shuffle_games = game_count * shuffle_rate
    
    env = gym.make("ChineseCheckers-v0")
    max_turns = 100

    # recording game states and ground truth
    states = []
    policy_ground_truths = []
    values = []
    actions = [] * game_count
    start_states = []

    for game in tqdm(range(game_count), desc="Generating games"):
        if game < random_move_games:
            shuffle_start = False
            random_move_game = True
        elif game < (shuffle_games + random_move_games):
            shuffle_start = True
            random_move_game = False
        else:
            shuffle_start = False
            random_move_game = False
        obs, info = env.reset(options={"shuffle_start": shuffle_start})

        if replay:
            start_states.append(info["board"].copy())

        game_turn = 0
        game_states = []
        game_truths = []
        game_turns = []
        game_actions = []
        terminated = False
        while not terminated:
            game_states.append(info["board"].to_tensor())
            best_actions = _get_best_actions(info["board"])
            truth = torch.zeros(size=[6, 7, 7])
            for action in best_actions:
                truth[action.piece_id, action.position.x, action.position.y] = 1 / len(best_actions)
            game_truths.append(truth)
            game_turns.append(info["turn"])

            if random_move_game and game_turn < len(agents) * random_turns:
                action = take_random_action(obs, info)
            else:
                agent_turn = info["turn"]
                action = agents[agent_turn].act(obs, info)
            # TODO: check for circular actions
            if replay:
                game_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            game_turn += 1

            if game_turn > max_turns:
                break
        
        if replay:
            actions.append(game_actions)

        if reward != 0:
            # game did not tie, we will use this training data
            winner = info["winner"]
            states.extend(game_states)
            policy_ground_truths.extend(game_truths)
            values.extend([-1 if winner == turn else 1 for turn in game_turns])

    data = list(zip(states, policy_ground_truths, values))
    if flip_double:
        data.extend(mirror_data(data))
    data = random.choices(data, k=int(keep*len(data)))
    if replay:
        replays = list(zip(start_states, actions))
        if flip_double:
            replays.extend([mirror_replay(replay) for replay in replays])
        if as_list:
            return data, replays
        else:
            return GameDataset(data), replays
    else:
        if as_list:
            return data
        else:
            return GameDataset(data)

def mirror_data(data: list) -> list:
    def mirror_state(state):
        new_state = -torch.ones_like(state, dtype=torch.float32)
        for x in range(len(state)):
            for y in range(len(state[x])):
                new_state[x][y] = state[6-y][6-x]
        return state
    
    def mirror_truth(truth):
        new_truth = -torch.ones_like(truth, dtype=torch.float32)
        for layer in range(len(truth)):
            for x in range(len(truth[layer])):
                for y in range(len(truth[layer][x])):
                    new_truth[layer][x][y] = truth[layer][6-y][6-x]
        return truth
    
    new_data = []

    for state, truth, value in data:
        new_data.append((mirror_state(state), mirror_truth(truth), value))

    return new_data

def mirror_replay(replay: tuple[Board, list[Action]]) -> tuple:
    start_state, actions = replay
    new_state = start_state.mirror()

    new_actions = []
    for action in actions:
        new_action = Action(piece_id=action.piece_id, position=Position(x=6-action.position.y, y=6-action.position.x))
        new_actions.append(new_action)

    return new_state, new_actions
    

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