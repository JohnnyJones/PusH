import gymnasium as gym
import random
import time
import numpy as np

from environment import ChineseCheckersEnv
from data import Action, Position
from agent import ChineseCheckersAgent, RandomAgent, DeterministicGreedyAgent, StochasticGreedyAgent, DeepMctsAgent
from tqdm import tqdm
from argparse import ArgumentParser
from collections import deque

def play(env: gym.Env, agents: list[ChineseCheckersAgent], agent_moves, agent_times) -> int:
    obs, info = env.reset()
    terminated = False
    game_turn = 0
    max_turns = 100
    random_turns_per_agent = 3
    action_history = {i: deque(maxlen=3) for i in range(len(agents))}
    while not terminated:
        if game_turn < len(agents) * random_turns_per_agent:
            action = take_random_action(obs, info)
        else:
            agent_turn = info["turn"]
            time_before = time.perf_counter()
            action = agents[agent_turn].act(obs, info)
            time_after = time.perf_counter()
            elapsed = time_after - time_before
            agent_moves[agent_turn] += 1
            agent_times[agent_turn] += elapsed
        # TODO: check for circular actions
        obs, reward, terminated, truncated, info = env.step(action)
        game_turn += 1
        if game_turn > max_turns:
            break
    
    return info["winner"]

def take_random_action(obs, info):
    actions = info["valid_actions_list"]
    if len(actions) == 0:
        raise ValueError("No valid actions available")
    return random.choice(actions)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int, default=1000)
    parser.add_argument("-r", "--render", type=str, default=None)
    parser.add_argument("-t", "--throw", action="store_true")
    args = parser.parse_args()

    env = gym.make("ChineseCheckers-v0", render_mode=args.render)
    agents: list[ChineseCheckersAgent] = [DeepMctsAgent(), DeterministicGreedyAgent()]
    episodes = args.episodes
    winners = []
    errors = 0
    ties = 0
    agent_turns = [0] * len(agents)
    agent_times = [0] * len(agents)

    if args.render is not None:
        r = range(episodes)
    else:
        r = tqdm(range(episodes))
    for i in r:
        if i % 2 == 1:
            agents = agents[::-1]
            agent_turns = agent_turns[::-1]
            agent_times = agent_times[::-1]
        try:
            winner = play(env, agents, agent_turns, agent_times)
            if winner != -1:
                winners.append(type(agents[winner]).__name__)
                if args.render is not None:
                    print(f"Winner: {type(agents[winner]).__name__}")
            else:
                ties += 1
        except Exception as e:
            if e == KeyboardInterrupt:
                break
            errors += 1
            if args.throw:
                raise e
            else:
                print(e)

    env.close()

    agent_names = [type(agent).__name__ for agent in agents]
    for agent in agent_names:
        print(f"{agent} wins : {winners.count(agent)}, {winners.count(agent)/episodes*100:.2f}%")
    for i in range(len(agents)):
        print(f"{agent_names[i]} average turn time : {agent_times[i] / agent_turns[i]:.3f} seconds")
    print(f"Ties: {ties}")
    print(f"Errors: {errors}")