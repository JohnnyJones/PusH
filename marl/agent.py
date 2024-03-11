import torch
import torch.nn as nn
import gymnasium as gym
from agentdqn import DQN
from collections import namedtuple
import random

class ChineseCheckersAgent():
    Action = namedtuple("Action", ["piece_id", "position"])
    Position = namedtuple("Position", ["x", "y"])

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
        pass
    
    def act(self, observation, info) -> ChineseCheckersAgent.Action:
        super(DeterministicGreedyAgent, self).act(observation, info)
        return self._get_best_action(observation, info)
    
    def _get_best_action(self, observation, info) -> ChineseCheckersAgent.Action:
        actions = info["valid_actions_list"]
        if len(actions) == 0:
            raise ValueError("No valid actions available")
        
        heuristics = [self._heuristic(observation, action, info) for action in actions]
        # find all the actions with the best heuristic
        best_heuristic = max(heuristics)
        best_actions = [action for action, heuristic in zip(actions, heuristics) if heuristic == best_heuristic]

        if len(best_actions) == 1:
            return best_actions[0]
        else:
            # we have multiple best actions
            # find furthest behind piece
            def distance(player, piece_id):
                position = ChineseCheckersAgent.Position(*info["id_to_position"][player][piece_id])
                distance = position.x - position.y
                if player == 1:
                    distance = -distance
                return distance

            # sort the best actions by the distance of the piece
            # index 0 will be the furthest behind piece
            best_piece_ids = [action.piece_id for action in best_actions]
            best_piece_ids.sort(key=lambda piece_id: distance(info["turn"], piece_id))

            # sample actions belonging to the best piece
            best_piece_actions = [action for action in best_actions if action.piece_id == best_piece_ids[0]]
            best_action = random.choice(best_piece_actions)

        return best_action

    
    def _heuristic(self, observation, action: ChineseCheckersAgent.Action, info):
        # Moving further is better
        turn = info["turn"]
        starting_position = ChineseCheckersAgent.Position(*info["id_to_position"][turn][action.piece_id])
        ending_position = action.position

        # due the ending positions being at the top right and bottom left of the board matrix,
        # higher x is better, lower y is better for player 0
        # lower x is better, higher y is better for player 1
        heuristic = (starting_position.x - starting_position.y) - (ending_position.x - ending_position.y)
        if turn == 1:
            heuristic = -heuristic
        return heuristic

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