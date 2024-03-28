import torch
import numpy as np
import random
import pandas as pd

from deepmctsmodel import DeepMctsModel
from collections import namedtuple
from data import Action, Position
from board import Board

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
        return self._choose_action(actions, starting_positions, heuristics, turn)
    
    def _choose_action(self, actions: list[Action], starting_positions: list[Position], heuristics: list[float], turn: int) -> Action:       
        df = self._get_best_actions(actions, starting_positions, heuristics, turn)

        # if there's only one action with the best heuristic, return it
        if len(df) == 1:
            return df.iloc[0]["action"]
        
        # filter by the worst starting position
        df = df[df["distance"] == df["distance"].max()]
        
        # sample from the best actions with the worst starting position
        return random.choice(df["action"].tolist())

    def _get_best_actions(self, actions: list[Action], starting_positions: list[Position], heuristics: list[float], turn: int) -> pd.DataFrame:       
        # join into a dataframe
        distances = [self._distance_to_goal(turn, start) for start in starting_positions]
        df = pd.DataFrame({"action": actions, "start": starting_positions, "heuristic": heuristics, "distance": distances})

        # filter by the best heuristic
        df = df.sort_values(by="heuristic", ascending=False)
        df = df[df["heuristic"] == df["heuristic"].max()]

        return df

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

    def _choose_action(self, actions: list[Action], starting_position: list[Position], heuristics: list[float], turn: int) -> Action:       
        normalized_heuristics = [heuristic * (heuristic > 0) for heuristic in heuristics]
        
        if sum(normalized_heuristics) == 0:
            # no positive heuristics, use the deterministic greedy agent
            return super()._choose_action(actions, starting_position, heuristics, turn)        
        
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
        self.parent: MctsTreeNode = None
        self.terminal: bool = False

    def __repr__(self):
        return f"[Node, ({self.visits}, {self.children})]"

    def add_child(self, child_node, action: Action):
        self.children.append(child_node)
        child_node.parent = self
        child_node.action = action
    
    def mean_value(self):
        if self.visits == 0.0:
            return 0.0
        return self.accumulated_value / self.visits

class DeepMctsAgent(ChineseCheckersAgent):
    def __init__(self, device: str = 'cpu') -> None:
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
        board: Board = info["board"]
        return self._get_best_action(board, temperature)
    
    def train_model(self, history):
        pass

    def _get_best_action(self, board: Board, temperature) -> Action:
        actions, visits = self._mcts(board, 180)
        try:
            action = self._decision(actions, visits, temperature)
        except Exception as e:
            print("Game and agent state leading to error")
            print("Board:")
            print(board)
            print("Actions")
            print(actions)
            print("Visits")
            print(visits)
            raise e
        return action

    def _mcts(self, board: Board, iterations: int) -> tuple[list[Action], list[int]]:
        root = MctsTreeNode(board)
        for _ in range(iterations):
            selection = self._selection(root, c=3.5)
            self._expansion(selection, board.turn)
            self._backup(selection, selection.parent)
        return [child.action for child in root.children], [child.visits for child in root.children]

    def _selection(self, node: MctsTreeNode, c: float = 3.5):
        if len(node.children) == 0:
            return node
        if node.terminal:
            # board is terminal
            return node
        total_child_visits = sum(child.visits for child in node.children)
        n = 0
        upper_confidence_bounds = np.array([child.mean_value() + c * child.prior_probability * 
                                    np.sqrt(total_child_visits) /(child.visits + 1) 
                                    for child in node.children])
        # continue with the child with the highest upper confidence bound     
        return self._selection(node.children[np.argmax(upper_confidence_bounds)], c)

    def _expansion(self, node: MctsTreeNode, turn: int):
        if node.board.check_win() != -1:
            node.terminal = True
            # use raw win vales
            if node.board.turn == turn:
                # you lost
                node.accumulated_value = -1
            else:
                # you won
                node.accumulated_value = +1
        else:
            # get priors and estimated value from nn
            with torch.inference_mode():
                value, prior = self.model(node.board.to_tensor())
            node.accumulated_value = value.item()

            # get valid actions
            valid_actions = node.board.get_valid_actions_list()

            # create children
            for action in valid_actions:
                new_board: Board = node.board.copy()
                new_board.move_piece(*action)
                new_node = MctsTreeNode(new_board)
                new_node.prior_probability = prior[0][action.piece_id, action.position.x, action.position.y]
                node.add_child(new_node, action)
        node.visits += 1

    def _backup(self, leaf: MctsTreeNode, node: MctsTreeNode):
        if node is None:
            return
        
        node.visits += 1

        # accumulate value
        if leaf.terminal:
            if node.board.turn == leaf.board.turn:
                node.accumulated_value -= leaf.accumulated_value
            else:
                node.accumulated_value += leaf.accumulated_value
        else:
            if node.board.turn == leaf.board.turn:
                node.accumulated_value += leaf.accumulated_value
            else:
                node.accumulated_value -= leaf.accumulated_value
        
        # continue with the parent
        if node.parent is not None:
            self._backup(leaf, node.parent)

    def _decision(self, actions, visits, temperature) -> Action:       
        if temperature == 0.0:
            return actions[np.argmax(visits)]
        
        # TODO: find out why visits sometimes are all zero
        probability_denominator = sum([visit_count ** (1 / temperature) for visit_count in visits])
        if probability_denominator == 0:
            if all([visit == 0 for visit in visits]):
                raise RuntimeError(f"Visit counts cannot all be zero")
            else:
                raise RuntimeError(f"Some other problem occured causing probability denominator to sum to zero")
        probabilities = [visit ** (1 / temperature) / probability_denominator for visit in visits]

        return random.choices(actions, weights=probabilities, k=1)[0]