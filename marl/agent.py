import torch
import torch.nn as nn
import gymnasium as gym
from agentdqn import DQN

# Chinese Checkers Agent
# Performs error checking
class ChineseCheckersAgent():
    def __init__(self):
        self.model = None

    def act(self, observation):
        # observation should be 7x7x7 tensor
        if not isinstance(observation, torch.Tensor):
            raise ValueError("observation should be a tensor")
        if observation.shape != (7, 7, 7):
            raise ValueError("observation should be a 7x7x7 tensor")
        
        return self._get_best_action(observation)

    def _get_best_action(self, observation):
        if self.model is None:
            raise ValueError("model not set")
 
class DeterministicGreedyAgent(ChineseCheckersAgent):
    def __init__(self) -> None:
        super(DeterministicGreedyAgent, self).__init__()

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