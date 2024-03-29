import gymnasium as gym
import random
import time
import numpy as np
import torch

from environment import ChineseCheckersEnv
from data import Action, Position
from agent import ChineseCheckersAgent, RandomAgent, DeterministicGreedyAgent, StochasticGreedyAgent, DeepMctsAgent
from tqdm import tqdm
from argparse import ArgumentParser
from collections import deque
from board import Board
from data_generation import game_generation

def replay(start_state: Board, actions: list[Action]):
    env = gym.make("ChineseCheckers-v0", render_mode="human", render_fps=2)
    env.reset(options={"start_state": start_state})

    for action in actions:
        env.step(action)

    env.close()

if __name__ == "__main__":
    mcts_agent = DeepMctsAgent()
    mcts_agent.model.load_state_dict(torch.load("./models/heuristic-trained-mcts-model-v0.pt"))
    dataset, replays = game_generation(1, [DeterministicGreedyAgent(), DeterministicGreedyAgent()], random_turns=6, random_move_rate=1.0, shuffle_rate=0.0, replay=True, flip_double=True)
    for game in replays:
        start_state, actions = game
        replay(start_state, actions)