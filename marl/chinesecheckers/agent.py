import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import random
import pandas as pd

from deepmctsmodel import DeepMctsModel
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
        actions: list[Action] = info["valid_actions_list"]
        turn = info["turn"]
        starting_positions = [Position(*info["id_to_position"][turn][action.piece_id]) for action in actions]
        if len(actions) == 0:
            raise ValueError("No valid actions available")
        heuristics = [self._heuristic(start, action.position, turn) for start, action in zip(starting_positions, actions)]
        return self._get_best_action(actions, starting_positions, heuristics, turn)
    
    def _get_best_action(self, actions: list[Action], starting_position: list[Position], heuristics: list[float], turn: int) -> Action:       
        # join into a dataframe
        distances = [self._distance_to_goal(turn, start) for start in starting_position]
        df = pd.DataFrame({"action": actions, "start": starting_position, "heuristic": heuristics, "distance": distances})

        # filter by the best heuristic
        df = df.sort_values(by="heuristic", ascending=False)
        df = df[df["heuristic"] == df["heuristic"].max()]

        # if there's only one action with the best heuristic, return it
        if len(df) == 1:
            return df.iloc[0]["action"]
        
        # filter by the worst starting position
        df = df[df["distance"] == df["distance"].max()]
        
        # sample from the best actions with the worst starting position
        return random.choice(df["action"].tolist())

    @staticmethod
    def _distance_to_goal(turn: int, position: Position):
        # due the ending positions being at the top right and bottom left of the board matrix,
        # higher x is better, lower y is better for player 0
        # lower x is better, higher y is better for player 1
        distance = position.x - position.y
        if turn == 1:
            distance = -distance
        return distance
    
    def _heuristic(self, start: Position, end: Position, turn) -> float:
        # Moving further is better
        heuristic = (self._distance_to_goal(turn, start)) - (self._distance_to_goal(turn, end))
        return heuristic

class StochasticGreedyAgent(DeterministicGreedyAgent):
    def __init__(self) -> None:
        super(StochasticGreedyAgent, self).__init__()

    def _get_best_action(self, actions: list[Action], starting_position: list[Position], heuristics: list[float], turn: int) -> Action:       
        normalized_heuristics = [heuristic * (heuristic > 0) for heuristic in heuristics]
        
        if sum(normalized_heuristics) == 0:
            # no positive heuristics, use the deterministic greedy agent
            return super()._get_best_action(actions, starting_position, heuristics, turn)        
        
        heuristic_sum = sum(normalized_heuristics)
        weights = [heuristic / heuristic_sum for heuristic in normalized_heuristics]
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

class DeepMctsAgent(ChineseCheckersAgent):
    def __init__(self, device: str = 'cpu') -> None:
        super(DeepMctsAgent, self).__init__()
        self.model = DeepMctsModel()
        self.device = torch.device(device)
        self.model.to(device)

        self._train = True

    def train(self):
        self._train = True
    
    def eval(self):
        self._train = False

    def act(self, observation, info) -> Action:
        super(DeepMctsAgent, self).act(observation, info)
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
            path = self._selection(root, c=3.5, out=node)
            self._expansion(node, info["turn"])
            self._backup(node, path)

    def _selection(self, node: MctsTreeNode, out: MctsTreeNode, c: float = 3.5, path: list[int] = []):
        if len(node.children) == 0:
            return path
        if node.board.check_win() != -1:
            # board is terminal
            return path
        total_child_visits = sum(child.visits for child in node.children)
        upper_confidence_bounds = [child.mean_value() + c * child.prior_probability * 
                                    np.sqrt(total_child_visits) /(child.visits +1) 
                                    for child in node.children]
        # get the index of the child with the highest upper confidence bound
        index = np.argmax(upper_confidence_bounds)
        path.append(self._selection(node.children[index], out, c))
        return path

    def _expansion(self, node: MctsTreeNode, turn: int):
        if node.board.check_win() == -1:
            # use raw win vales
            if node.board.turn == turn:
                # you lost
                node.accumulated_value -= 1
            else:
                # you won
                node.accumulated_value += 1
        else:
            # get priors and estimated value from nn
            value, prior = self.model()

            # get valid actions
            valid_actions = node.board.get_valid_actions()

            # create children
            for action in valid_actions:
                new_board = node.board.copy()
                new_board.move(action)
                new_node = MctsTreeNode(new_board)
                new_node.prior_probability = prior[action]
                node.add_child(new_node)

    def _backup(self, node: MctsTreeNode, path: list[int]):
        for index in path:
            node = node.children[index]
            node.visits += 1
            node.accumulated_value = node.mean_value()

    def _decision(self, actions, visits, temperature) -> Action:
        if temperature == 0.0:
            return actions[np.argmax(visits)]
        
        probability_denominator = sum(visits ** (1 / temperature))
        probabilites = [visit ** (1 / temperature) / probability_denominator for visit in visits]

        return random.choices(actions, weights=probabilites, k=1)[0]