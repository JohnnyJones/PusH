import numpy as np
import pygame
import gymnasium as gym

from collections import namedtuple
from gymnasium import spaces
from gymnasium.envs.registration import register

class ChineseCheckersEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
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

        # game board is 7x7 matrix (diamond-shaped board) with 2 players
        def __init__(self):
            self.reset()

        def __repr__(self) -> str:
            return str(self.board+1)
        
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
            # clear the previous position
            from_position = self.Position(self.id_to_position[player_id, piece_id])
            self.board[from_position] = 0
            self.position_to_id[player_id, from_position.x, from_position.y] = -1

            # set the new position
            self.board[to_position] = player_id
            self.position_to_id[player_id, to_position.x, to_position.y] = piece_id
        
        def check_win(self):
            # top right of piece matrix
            p1_win_positions = [
                (0, 4), (0, 5), (0, 6),
                        (1, 5), (1, 6),
                                (2, 6),
            ]

            # bottom left of piece matrix
            p2_win_positions = [
                (4, 0),
                (5, 0), (5, 1),
                (6, 0), (6, 1), (6, 2),
            ]

            if all([self.position_to_id[i][1] == 1 for i in p1_win_positions]):
                return 1
            elif all([self.position_to_id[i][1] == 2 for i in p2_win_positions]):
                return 2
            return 0


    def __init__(self, render_mode=None):
        self.window_size = 512 # Size of the PyGame window

        # observation space is 2 players, 7x7 board, 6 pieces
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([1, 6, 6, 5]), dtype=np.int8)

        # action space is 6 pieces, 7x7 board destinations
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([5, 6, 6]), dtype=np.int8)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.board = self.Board()
        print(self.board)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.COLORS["BACKGROUND"])

    def _get_info(self):
        return {}

    def _get_obs(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        reward = 0
        terminated = False
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

register(
    id="ChineseCheckers-v0",
    entry_point="chinesecheckers:ChineseCheckersEnv",
)