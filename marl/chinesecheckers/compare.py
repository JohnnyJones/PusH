import gymnasium as gym
import random

from chinesecheckers import ChineseCheckersEnv
from data import Action, Position
from agent import ChineseCheckersAgent, RandomAgent, DeterministicGreedyAgent, StochasticGreedyAgent
from tqdm import tqdm
from argparse import ArgumentParser
from collections import deque

def play(env: gym.Env, agents: list[ChineseCheckersAgent]) -> int:
    obs, info = env.reset()
    terminated = False
    turn = 0
    max_turns = 100
    random_turns_per_agent = 3
    action_history = {i: deque(maxlen=3) for i in range(len(agents))}
    while not terminated:
        if turn < len(agents) * random_turns_per_agent:
            action = take_random_action(obs, info)
        else:
            action = agents[info["turn"]].act(obs, info)
        # TODO: check for circular actions
        obs, reward, terminated, truncated, info = env.step(action)
        turn += 1
        if turn > max_turns:
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
    args = parser.parse_args()

    env = gym.make("ChineseCheckers-v0", render_mode=args.render)
    agents: list[ChineseCheckersAgent] = [StochasticGreedyAgent(), DeterministicGreedyAgent()]
    episodes = args.episodes
    winners = []
    errors = 0
    ties = 0

    if args.render is not None:
        r = range(episodes)
    else:
        r = tqdm(range(episodes))
    for i in r:
        if i % 2 == 0:
            agents = agents[::-1]
        try:
            winner = play(env, agents)
            if winner != -1:
                winners.append(type(agents[winner]).__name__)
            else:
                ties += 1
        except Exception as e:
            if e == KeyboardInterrupt:
                break
            errors += 1
            print(e)

    env.close()

    agent_names = [type(agent).__name__ for agent in agents]
    for agent in agent_names:
        print(f"{agent} wins : {winners.count(agent)}, {winners.count(agent)/episodes*100:.2f}%")
    print(f"Ties: {ties}")
    print(f"Errors: {errors}")