from chinesecheckers import ChineseCheckersEnv
from collections import namedtuple
from agent import ChineseCheckersAgent, RandomAgent, DeterministicGreedyAgent, StochasticGreedyAgent
import gymnasium as gym
from tqdm import tqdm

Action = namedtuple("Action", ["piece_id", "position"])

def play(env: gym.Env, agents: ChineseCheckersAgent) -> int:
    obs, info = env.reset()
    terminated = False
    while not terminated:
        action = agents[info["turn"]].act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

    return info["winner"]

if __name__ == "__main__":
    env = gym.make("ChineseCheckers-v0")
    agents = [StochasticGreedyAgent(), DeterministicGreedyAgent()]
    episodes = 20000
    winners = []
    errors = 0
    ties = 0
    for i in tqdm(range(episodes)):
        try:
            winner = play(env, agents)
            if winner != -1:
                winners.append(winner)
            else:
                ties += 1
        except:
            errors += 1

    env.close()

    for i, agent in enumerate(agents):
        print(f"{type(agent).__name__} wins : {winners.count(i)}, {winners.count(i)/episodes*100:.2f}%")
    print(f"Ties: {ties}")
    print(f"Errors: {errors}")