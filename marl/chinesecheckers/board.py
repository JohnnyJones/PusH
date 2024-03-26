import numpy as np
import torch

from data import Action, Position
from collections import deque


class Board: 
    # game board is 7x7 matrix (diamond-shaped board) with 2 players
    def __init__(self, start=None):
        self.turn = 0
        self.history = deque(maxlen=3)
        if start:
            self.position_to_id = start
            self.board = self._position_matrix_to_board_matrix(start)
            self.id_to_position = self._position_matrix_to_id_matrix(start)
        else:
            self.reset()

    @staticmethod
    def _position_matrix_to_board_matrix(mat: np.ndarray):
        board = -np.ones((7, 7), dtype=np.int8)
        for i in range(mat.shape[0]):
            sub_board = mat[i]
            sub_board[sub_board > -1] = i
            board = np.maximum(board, sub_board)
        return board

    @staticmethod
    def _position_matrix_to_id_matrix(mat: np.ndarray):
        id_to_position = -np.ones((mat.shape[0], 6, 2), dtype=np.int8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                for k in range(mat.shape[2]):
                    if mat[i, j, k] > -1:
                        id_to_position[i, mat[i, j, k]] = (j, k)
        return id_to_position

    def __repr__(self) -> str:
        board_string = []
        for row in self.board:
            board_string.append("[ ")
            for cell in row:
                if cell == -1:
                    board_string.append(". ")
                else:
                    board_string.append(str(cell) + " ")
            board_string.append("]\n")
        return "".join(board_string)
    
    def starting_positions(self, player_id):
        p0_starting_positions = [
            (4, 0),
            (5, 0), (5, 1),
            (6, 0), (6, 1), (6, 2),
        ]

        p1_starting_positions = [
            (0, 4), (0, 5), (0, 6),
                    (1, 5), (1, 6),
                            (2, 6),
        ]

        return p0_starting_positions if player_id == 0 else p1_starting_positions
    
    def win_positions(self, player_id):
        return self.starting_positions(1 - player_id)

    def reset(self):
        self.board = -np.ones((7, 7), dtype=np.int8) # 7 rows, 7 columns, output is player_id
        self.position_to_id = -np.ones((2, 7, 7), dtype=np.int8) # 2 player_ids, 7 rows, 7 columns, output is piece_id
        self.id_to_position = -np.ones((2, 6, 2), dtype=np.int8) # 2 player_ids, 6 piece_ids, 2 coordinates, output is (x, y)

        for player_id in range(2):
            for i, position in enumerate(self.starting_positions(player_id)):
                self.add_piece(player_id, i, Position(*position))
        
        self.turn = 0
        self.history.clear()
        for _ in range(3):
            self.history.appendleft(self.position_to_id)
    
    def copy(self):
        new_board = Board()
        new_board.board = self.board.copy()
        new_board.position_to_id = self.position_to_id.copy()
        new_board.id_to_position = self.id_to_position.copy()
        new_board.history = self.history.copy()
        new_board.turn = self.turn
        return new_board

    def to_tensor(self) -> torch.Tensor:        
        if (n := len(self.history)) != 3:
            raise ValueError(f"History must have 3 states, got {n}")

        history = []
        # concatenate history into single ndarray
        if self.turn == 0:
            # stack as is
            for state in self.history:
                history += [*state]
            history = np.stack(history)
        else:
            # reverse the order of each state
            reverse_history = [state[::-1] for state in self.history]
            for state in reverse_history:
                history += [*state]
            history = np.stack(history)

        # add layer for player turn
        history = np.concatenate((history, np.ones((1, 7, 7), dtype=np.int8) * self.turn)).astype(dtype=np.float32, copy=False)

        return torch.from_numpy(history).detach()

    def add_piece(self, player_id, piece_id, position: Position):
        self.board[position] = player_id
        self.position_to_id[player_id, position.x, position.y] = piece_id
        self.id_to_position[player_id, piece_id] = position

    def move_piece(self, piece_id, to_position: Position, player_id=None):
        if player_id is None:
            player_id = self.turn
        
        # convert to Position object
        to_position = Position(*to_position)
        
        # clear the previous position
        from_position = Position(*self.id_to_position[player_id, piece_id])
        self.board[from_position] = -1
        self.position_to_id[player_id, from_position.x, from_position.y] = -1

        # set the new position
        self.board[to_position] = player_id
        self.position_to_id[player_id, to_position.x, to_position.y] = piece_id
        self.id_to_position[player_id, piece_id] = to_position

        self.turn = 1 - self.turn
        
        # add current state to game history
        self.history.appendleft(self.position_to_id)
    
    def check_win(self):
        for player_id in range(2):
            win_positions = self.win_positions(player_id)

            # all win positions must be occupied and winning player must have pieces in the win positions
            if any([self.board[i] == player_id for i in win_positions]) and all([self.board[i] != -1 for i in win_positions]):
                return player_id
        return -1

    def get_action_mask(self, player_id=None):
        if player_id is None:
            player_id = self.turn
        
        moves = np.zeros((6, 7, 7), dtype=np.bool_)
        
        # rolling
        for piece_id in range(6):
            # check adjacent positions
            position = Position(*self.id_to_position[player_id, piece_id])
            adjacent_positions = self.get_reachable_positions(position)
            for adjacent_position in adjacent_positions:
                moves[piece_id, adjacent_position.x, adjacent_position.y] = True

        return moves
    
    def get_valid_actions_list(self, player_id=None) -> list[Action]:
        if player_id is None:
            player_id = self.turn
        
        actions = []
        for piece_id in range(6):
            position = Position(*self.id_to_position[player_id, piece_id])
            reachable_positions = self.get_reachable_positions(position)
            for reachable_position in reachable_positions:
                action = Action(piece_id, 
                                        Position(reachable_position.x, reachable_position.y))
                actions.append(action)
        return actions

    def get_valid_actions_dict(self, player_id=None) -> dict[int, list[Position]]:
        if player_id is None:
            player_id = self.turn
        
        actions = {}
        for piece_id in range(6):
            position = Position(*self.id_to_position[player_id, piece_id])
            reachable_positions = self.get_reachable_positions(position)
            for reachable_position in reachable_positions:
                if piece_id not in actions:
                    actions[piece_id] = [reachable_position]
                else:
                    actions[piece_id].append(reachable_position)
        return actions

    def get_adjacent_positions(self, position: Position) -> list[Position]:
        positions = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx + dy == 0:
                    continue
                new_position = Position(position.x + dx, position.y + dy)
                if 0 <= new_position.x < 7 and 0 <= new_position.y < 7:
                    positions.append(new_position)
    
    def get_unoccupied_adjacent_positions(self, position: Position) -> list[Position]:
        positions = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx + dy == 0:
                    continue
                new_position = Position(position.x + dx, position.y + dy)
                if 0 <= new_position.x < 7 and 0 <= new_position.y < 7 and self.board[new_position] == -1:
                    positions.append(new_position)
        return positions
    
    def get_occupied_adjacent_positions(self, position: Position) -> list[Position]:
        positions = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx + dy == 0:
                    continue
                new_position = Position(position.x + dx, position.y + dy)
                if 0 <= new_position.x < 7 and 0 <= new_position.y < 7 and self.board[new_position] != -1:
                    positions.append(new_position)
        return positions
    
    def get_hop_positions(self, position: Position) -> list[Position]:
        positions = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx + dy == 0:
                    continue
                adj_position = Position(position.x + dx, position.y + dy)
                hop_position = Position(position.x + 2 * dx, position.y + 2 * dy)
                if (0 <= hop_position.x < 7 and 0 <= hop_position.y < 7 and
                    self.board[adj_position] != -1):
                    positions.append(hop_position)
        return positions

    def get_unoccupied_hop_positions(self, position: Position) -> list[Position]:
        positions = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx + dy == 0:
                    continue
                adj_position = Position(position.x + dx, position.y + dy)
                hop_position = Position(position.x + 2 * dx, position.y + 2 * dy)
                if (0 <= hop_position.x < 7 and 0 <= hop_position.y < 7 and
                    self.board[adj_position] != -1 and self.board[hop_position] == -1):
                    positions.append(hop_position)
        return positions
    
    def get_reachable_positions(self, position: Position) -> list[Position]:
        positions = []
        
        # recursive hopping using depth-first search
        def hop_dfs(position, visited: set = None, stack: list = None):
            if visited is None:
                visited = set()
            if stack is None:
                stack = []
            
            reachable = []

            # can be multiple ways to reach same position
            if position in visited:
                return reachable
            visited.add(position)
            
            neighbors = self.get_unoccupied_hop_positions(position)
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
                    reachable.append(neighbor)
            
            while stack:
                new_position = stack.pop()
                reachable.extend(hop_dfs(new_position, visited, stack))

            return reachable

        # hopping positions
        positions.extend(hop_dfs(position))

        # rolling positions
        positions.extend(self.get_unoccupied_adjacent_positions(position))
        
        return positions
    
    def observation(self):
        return self.position_to_id
