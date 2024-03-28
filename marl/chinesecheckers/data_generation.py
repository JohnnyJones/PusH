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

def game_generation(game_count, agents: list[ChineseCheckersAgent] = [DeterministicGreedyAgent(), DeterministicGreedyAgent()], random_turns=0, shuffle=False, as_list=True):
    env = gym.make("ChineseCheckers-v0")
    obs, info = env.reset()
    max_turns = 100

    # recording game states and ground truth
    states = []
    policy_ground_truths = []
    values = []

    for _ in tqdm(range(game_count), desc="Generating games"):        
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
            states.extend(game_states)
            policy_ground_truths.extend(game_truths)
            values.extend([-1 if winner == turn else 1 for turn in game_turns])
    
    data = list(zip(states, policy_ground_truths, values))
    if as_list:
        return data
    else:
        raise NotImplementedError()
            

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