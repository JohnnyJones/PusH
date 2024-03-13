import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import random

from agentdqn import DQN
from collections import namedtuple
from data import Action, Position
from chinesecheckers import Board, ChineseCheckersEnv

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

    def act(self, observation, info: dict) -> Action:
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
    
    def act(self, observation, info) -> Action:
        super(DeterministicGreedyAgent, self).act(observation, info)
        actions = info["valid_actions_list"]
        if len(actions) == 0:
            raise ValueError("No valid actions available")
        heuristics = [self._heuristic(observation, action, info) for action in actions]
        return self._get_best_action(actions, heuristics, info)
    
    def _get_best_action(self, actions: list[Action], heuristics: list, info) -> Action:       
        # find all the actions with the best heuristic
        best_heuristic = max(heuristics)
        best_actions = [action for action, heuristic in zip(actions, heuristics) if heuristic == best_heuristic]

        if len(best_actions) == 1:
            return best_actions[0]
        else:
            # we have multiple best actions
            # find furthest behind piece
            def distance(player, piece_id):
                position = Position(*info["id_to_position"][player][piece_id])
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

    
    def _heuristic(self, observation, action: Action, info):
        # Moving further is better
        turn = info["turn"]
        starting_position = Position(*info["id_to_position"][turn][action.piece_id])
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

    def _get_best_action(self, actions: list[Action], heuristics: list, info) -> Action:       
        normalized_heuristics = [heuristic * (heuristic > 0) for heuristic in heuristics]
        
        if sum(normalized_heuristics) == 0:
            # no positive heuristics, use the deterministic greedy agent
            return super(StochasticGreedyAgent, self)._get_best_action(actions, heuristics, info)        
        
        weights = [heuristic / sum(normalized_heuristics) for heuristic in normalized_heuristics]
        action = random.choices(actions, weights=weights, k=1)[0]

        return action

class MctsTreeNode:
    def __init__(self, board: Board):
        self.board: Board = board
        self.children: list[MctsTreeNode] = []
        self.visits: int = 0
        self.accumulated_value: float = 0.0
        self.prior_probability: float = 0.0

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def remove_child(self, child_node):
        self.children.remove(child_node)
    
    def mean_value(self):
        return self.accumulated_value / self.visits

class DQNAgent(ChineseCheckersAgent):
    def __init__(self, device: str = 'cpu') -> None:
        super(DQNAgent, self).__init__()
        self.model = DQN()
        self.device = torch.device(device)
        self.model.to(device)

        self._train = True

    def train(self):
        self._train = True
    
    def eval(self):
        self._train = False

    def act(self, observation, info) -> Action:
        super(DQNAgent, self).act(observation, info)
        if self._train:
            temperature = 1.2
        else:
            temperature = 0.0
        return self._get_best_action(observation, info, temperature)
    
    def _get_best_action(self, observation, info, temperature) -> Action:
        actions, visits = self._mcts(observation, info, 100)
        return self._decision(self, actions, visits, temperature)

    def _mcts(self, observation, info, iterations) -> tuple[list[Action], list[int]]:
        root = MctsTreeNode(Board(observation))
        for i in range(iterations):
            node: MctsTreeNode
            self._selection(root, c=3.5, out=node)
            self._expansion(node, info["turn"])

    def _selection(self, node: MctsTreeNode, out: MctsTreeNode, c: float = 3.5):
        if len(node.children) == 0:
            return node
        if node.board.check_win() != -1:
            # board is terminal
            return node
        total_child_visits = sum(child.visits for child in node.children)
        upper_confidence_bounds = [child.mean_value() + c * child.prior_probability * 
                                    np.sqrt(total_child_visits) /(child.visits +1) 
                                    for child in node.children]
        return self._selection(node.children[np.argmax(upper_confidence_bounds)], out, c)

    def _expansion(self, node: MctsTreeNode, turn: int):
        if node.board.check_win() == -1:
            # use raw win vales
            if node.board.turn == turn:
                # you lost
                node.accumulated_value -= 1
            else:
                # you won
                node.accumulated_value += 1
        # else:
        #     # get priors and estimated value from nn
        #     prior, value = self.model()

    def _backup(self, observation):
        pass

    def _decision(self, actions, visits, temperature) -> Action:
        if temperature == 0.0:
            return actions[np.argmax(visits)]
        
        probability_denominator = sum(visits ** (1 / temperature))
        probabilites = [visit ** (1 / temperature) / probability_denominator for visit in visits]

        return random.choices(actions, weights=probabilites, k=1)[0]