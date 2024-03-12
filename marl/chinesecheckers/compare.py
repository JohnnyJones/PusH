import gymnasium as gym

from chinesecheckers import ChineseCheckersEnv
from data import Action, Position
from agent import ChineseCheckersAgent, RandomAgent, DeterministicGreedyAgent, StochasticGreedyAgent
from tqdm import tqdm
from argparse import ArgumentParser


def play(env: gym.Env, agents: ChineseCheckersAgent) -> int:
    obs, info = env.reset()
    terminated = False
    while not terminated:
        action = agents[info["turn"]].act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

    return info["winner"]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int, default=1000)
    parser.add_argument("-r", "--render", type=str, default=None)
    args = parser.parse_args()

    env = gym.make("ChineseCheckers-v0", render_mode=args.render)
    agents = [StochasticGreedyAgent(), DeterministicGreedyAgent()]
    episodes = args.episodes
    winners = []
    errors = 0
    ties = 0

    if args.render is not None:
        r = range(episodes)
    else:
        r = tqdm(range(episodes))
    for i in r:
        try:
            winner = play(env, agents)
            if winner != -1:
                winners.append(winner)
            else:
                ties += 1
        except Exception as e:
            if e == KeyboardInterrupt:
                break
            errors += 1
            print(e)

    env.close()

    for i, agent in enumerate(agents):
        print(f"{type(agent).__name__} wins : {winners.count(i)}, {winners.count(i)/episodes*100:.2f}%")
    print(f"Ties: {ties}")
    print(f"Errors: {errors}")