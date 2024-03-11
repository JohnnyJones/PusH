import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import numpy as np
import pygame
import gymnasium as gym

from collections import namedtuple
from gymnasium import spaces
from gymnasium.envs.registration import register

class ChineseCheckersEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "terminal"],
    }

    COLORS = {
        "BACKGROUND": (255, 255, 255),
        "EMPTY":      (139, 139, 139),
        "P1":         (0,   0,   0  ),
        "P2":         (255, 255, 255),
        "P3":         (255, 0,   0  ),
        "P4":         (0,   0,   255),
        "P5":         (0,   255, 0  ),
        "P6":         (255, 255, 0  ),
    }

    class Board: 
        class Position(namedtuple("Position", ["x", "y"])): pass
        class Action(namedtuple("Action", ["piece_id", "position"])): pass

        # game board is 7x7 matrix (diamond-shaped board) with 2 players
        def __init__(self):
            self.reset()

        def __repr__(self) -> str:
            return str(self.board)
        
        def reset(self):
            self.board = -np.ones((7, 7), dtype=np.int8) # 7 rows, 7 columns, output is player_id
            self.position_to_id = -np.ones((2, 7, 7), dtype=np.int8) # 2 player_ids, 7 rows, 7 columns, output is piece_id
            self.id_to_position = -np.ones((2, 6, 2), dtype=np.int8) # 2 player_ids, 6 piece_ids, 2 coordinates, output is (x, y)

            # P0 starting positions
            p0_starting_positions = [
                (4, 0),
                (5, 0), (5, 1),
                (6, 0), (6, 1), (6, 2),
            ]
            
            for i, position in enumerate(p0_starting_positions):
                self.add_piece(0, i, self.Position(*position))

            # P1 starting positions
            p1_starting_positions = [
                (0, 4), (0, 5), (0, 6),
                        (1, 5), (1, 6),
                                (2, 6),
            ]

            for i, position in enumerate(p1_starting_positions):
                self.add_piece(1, i, self.Position(*position))
        
        def add_piece(self, player_id, piece_id, position: Position):
            self.board[position] = player_id
            self.position_to_id[player_id, position.x, position.y] = piece_id
            self.id_to_position[player_id, piece_id] = position

        def move_piece(self, player_id, piece_id, to_position: Position):
            # convert to Position object
            to_position = self.Position(*to_position)
            
            # clear the previous position
            from_position = self.Position(*self.id_to_position[player_id, piece_id])
            self.board[from_position] = -1
            self.position_to_id[player_id, from_position.x, from_position.y] = -1

            # set the new position
            self.board[to_position] = player_id
            self.position_to_id[player_id, to_position.x, to_position.y] = piece_id
            self.id_to_position[player_id, piece_id] = to_position
        
        def check_win(self):
            # top right of piece matrix
            p0_win_positions = [
                (0, 4), (0, 5), (0, 6),
                        (1, 5), (1, 6),
                                (2, 6),
            ]

            # bottom left of piece matrix
            p1_win_positions = [
                (4, 0),
                (5, 0), (5, 1),
                (6, 0), (6, 1), (6, 2),
            ]

            if all([self.board[i] == 0 for i in p0_win_positions]):
                return 0
            elif all([self.board[i] == 1 for i in p1_win_positions]):
                return 1
            return -1

        def get_action_mask(self, player_id):
            moves = np.zeros((6, 7, 7), dtype=np.bool_)
            
            # rolling
            for piece_id in range(6):
                # check adjacent positions
                position = self.Position(*self.id_to_position[player_id, piece_id])
                adjacent_positions = self.get_reachable_positions(position)
                for adjacent_position in adjacent_positions:
                    moves[piece_id, adjacent_position.x, adjacent_position.y] = True

            return moves
        
        def get_valid_actions_list(self, player_id):
            actions = []
            for piece_id in range(6):
                position = self.Position(*self.id_to_position[player_id, piece_id])
                reachable_positions = self.get_reachable_positions(position)
                for reachable_position in reachable_positions:
                    action = self.Action(piece_id, 
                                         self.Position(reachable_position.x, reachable_position.y))
                    actions.append(action)
            return actions

        def get_valid_actions_dict(self, player_id):
            actions = {}
            for piece_id in range(6):
                position = self.Position(*self.id_to_position[player_id, piece_id])
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
                    new_position = self.Position(position.x + dx, position.y + dy)
                    if 0 <= new_position.x < 7 and 0 <= new_position.y < 7:
                        positions.append(new_position)
        
        def get_unoccupied_adjacent_positions(self, position: Position) -> list[Position]:
            positions = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx + dy == 0:
                        continue
                    new_position = self.Position(position.x + dx, position.y + dy)
                    if 0 <= new_position.x < 7 and 0 <= new_position.y < 7 and self.board[new_position] == -1:
                        positions.append(new_position)
            return positions
        
        def get_occupied_adjacent_positions(self, position: Position) -> list[Position]:
            positions = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx + dy == 0:
                        continue
                    new_position = self.Position(position.x + dx, position.y + dy)
                    if 0 <= new_position.x < 7 and 0 <= new_position.y < 7 and self.board[new_position] != -1:
                        positions.append(new_position)
            return positions
        
        def get_hop_positions(self, position: Position) -> list[Position]:
            positions = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx + dy == 0:
                        continue
                    adj_position = self.Position(position.x + dx, position.y + dy)
                    hop_position = self.Position(position.x + 2 * dx, position.y + 2 * dy)
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
                    adj_position = self.Position(position.x + dx, position.y + dy)
                    hop_position = self.Position(position.x + 2 * dx, position.y + 2 * dy)
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


    def __init__(self, render_mode=None):
        self.window_size = 512 # Size of the PyGame window

        # observation space is 2 players, 7x7 board
        self.observation_space = spaces.Box(low=-1, high=6, shape=(2, 7, 7), dtype=np.int8)

        # action space is 6 pieces, 7x7 board destinations
        self.action_space = spaces.MultiDiscrete([6, 7, 7])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.board = self.Board()
        self.turn = 0

    def _render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "terminal":
            print(self.board)
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        
        # TODO: render the board
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.COLORS["BACKGROUND"])
    
    def _get_obs(self):
        return self.board.observation()

    def _get_info(self):
        return {
            "turn": f"P{self.turn}",
            "action_mask": self.board.get_action_mask(self.turn),
            "valid_actions_dict": self.board.get_valid_actions_dict(self.turn),
            "valid_actions_list": self.board.get_valid_actions_list(self.turn),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        self._render()
        
        return observation, info
    
    def step(self, action):
        # TODO: check if action is valid
        piece_id, move_position = action
        move_position = self.board.Position(*move_position)
        valid_actions = self.board.get_action_mask(self.turn)
        if not valid_actions[piece_id, move_position.x, move_position.y]:
            raise ValueError("Invalid action")
        self.board.move_piece(*(self.turn, *action))

        reward = 0
        winner = self.board.check_win()
        if winner != -1:
            reward = 1 if winner == self.turn else -1

        self.turn = 1 - self.turn
        terminated = reward != 0
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        self._render()

        return observation, reward, terminated, truncated, info
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

register(
    id="ChineseCheckers-v0",
    entry_point="chinesecheckers:ChineseCheckersEnv",
)