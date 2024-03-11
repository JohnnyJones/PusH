import torch
import torch.nn as nn
import gymnasium as gym
from agentdqn import DQN
from collections import namedtuple
import random

class ChineseCheckersAgent():
    Action = namedtuple("Action", ["piece_id", "position"])

    def __init__(self) -> None:
        raise NotImplementedError("ChineseCheckersAgent is an abstract class")

    def act(self, observation, info) -> Action:
        if observation is None:
            raise ValueError("Observation cannot be None")
        if info is None:
            raise ValueError("Info cannot be None")
 
class RandomAgent(ChineseCheckersAgent):
    def __init__(self) -> None:
        return

    def act(self, observation, info: dict) -> ChineseCheckersAgent.Action:
        super(RandomAgent, self).act(observation, info)
        return self._get_random_action(observation, info)
    
    def _get_random_action(self, observation, info: dict):
        actions = info["valid_actions_list"]
        if len(actions) == 0:
            raise ValueError("No valid actions available")
        return random.choice(actions)

class DeterministicGreedyAgent(ChineseCheckersAgent):
    def __init__(self) -> None:
        super(DeterministicGreedyAgent, self).__init__()
    
    def act(self, observation):
        super(DeterministicGreedyAgent, self).act(observation)
        return self._get_best_action(observation)
    
    def _get_best_action(self, observation):
        pass

class StochasticGreedyAgent(DeterministicGreedyAgent):
    def __init__(self) -> None:
        super(StochasticGreedyAgent, self).__init__()

class DQNAgent(ChineseCheckersAgent):
    def __init__(self, device: str = 'cpu') -> None:
        super(DQNAgent, self).__init__()
        self.model = DQN()
        self.device = torch.device(device)
        self.model.to(device)

    def act(self, observation):
        super(DQNAgent, self).act(observation)
        return self._get_best_action(observation)