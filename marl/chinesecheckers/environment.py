import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import numpy as np
import pygame
import gymnasium as gym

from board import Board
from gymnasium import spaces
from gymnasium.envs.registration import register
from data import Action, Position

class ChineseCheckersEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "terminal"],
        "render_fps": 16,
    }

    COLORS = {
        "BACKGROUND": (56,  30,  26 ),
        "EMPTY":      (81,  56,  46 ),
        "HIGHLIGHT":  (100, 100, 100),
        "P0":         (0,   0,   0  ),
        "P1":         (255, 255, 255),
        "P2":         (255, 0,   0  ),
        "P3":         (0,   0,   255),
        "P4":         (0,   255, 0  ),
        "P5":         (255, 255, 0  ),
    }

    def __init__(self, render_mode=None):
        self.window_size = 1024 # Size of the PyGame window

        # observation space is 2 players, 7x7 board
        self.observation_space = spaces.Box(low=-1, high=6, shape=(2, 7, 7), dtype=np.int8)

        # action space is 6 pieces, 7x7 board destinations
        self.action_space = spaces.MultiDiscrete([6, 7, 7])

        self.last_action = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.board = Board()

    def _render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "terminal":
            self._render_terminal()
    
    def _render_terminal(self):
        print(self.board)
    
    def _render_frame(self, selection=False):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill(self.COLORS["BACKGROUND"])

        # draw the board diamond
        piece_radius = self.window_size // 14 // 2
        piece_size = piece_radius * 2

        def piece_position(x, y):
            starting_position = (self.window_size // 2 - piece_size * 3, self.window_size // 2)
            position_adjustment = (0.5*(x + y) * piece_size, (x - y) * piece_size)
            position = (starting_position[0] + position_adjustment[0], starting_position[1] + position_adjustment[1])
            return position

        # draw the starting positions
        for player_id in range(2):
            for i, position in enumerate(self.board.starting_positions(player_id)):
                color = self.mute_color(self.COLORS[f"P{player_id}"])
                pygame.draw.circle(
                    self.window,
                    color,
                    piece_position(*position),
                    piece_radius,
                    0
                )

        # draw the pieces
        for x in range(7):
            for y in range(7):
                position = piece_position(x, y)
                color = self.COLORS["EMPTY"]
                width = piece_radius // 5
                if self.board.board[x, y] != -1:
                    color = self.COLORS[f"P{self.board.board[x, y]}"]
                    width = 0
                
                # draw the pieces
                pygame.draw.circle(
                    self.window,
                    color,
                    position,
                    piece_radius,
                    width
                )

                # draw the player id
                if self.board.board[x, y] != -1:
                    font = pygame.font.Font(None, 36)
                    text = font.render(str(self.board.board[x, y]), True, self.invert_color(color))
                    text_rect = text.get_rect(center=position)
                    self.window.blit(text, text_rect)
        
        if selection:
            # draw the valid actions for the selected piece
            if self.last_action is not None:
                piece_id, move_position = self.last_action
                move_position = Position(*move_position)
                position = Position(*self.board.id_to_position[self.board.turn, piece_id])
                valid_actions = self.board.get_valid_actions_dict()[piece_id]
                # highlight the selected piece
                pygame.draw.circle(
                    self.window,
                    self.COLORS["HIGHLIGHT"],
                    piece_position(*position),
                    piece_radius,
                    5
                )
                # highlight the valid actions
                for valid_action in valid_actions:
                    pygame.draw.circle(
                        self.window,
                        self.COLORS["HIGHLIGHT"],
                        piece_position(*valid_action),
                        piece_radius,
                        5
                    )

        # render the frame
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_selection(self):
        self._render_frame(selection=True)
    
    def invert_color(self, color):
        return tuple(255 - c for c in color)
    
    def mute_color(self, color):
        # bring close to grey
        return tuple((c + 100) // 2 for c in color)

    def _get_obs(self):
        return self.board.observation()

    def _get_info(self):
        return {
            "turn": self.board.turn,
            "action_mask": self.board.get_action_mask(),
            "valid_actions_dict": self.board.get_valid_actions_dict(),
            "valid_actions_list": self.board.get_valid_actions_list(),
            "id_to_position": self.board.id_to_position,
            "winner": self.board.check_win(),
            "board": self.board.copy(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset(shuffle=options["shuffle_start"])

        observation = self._get_obs()
        info = self._get_info()

        self._render()
        
        return observation, info
    
    def step(self, action: Action):
        if action not in self.board.get_valid_actions_list():
            raise ValueError(f"Invalid action: {action}")
        self.last_action = action
        if self.render_mode == "human":
            self._render_selection()
        self.board.move_piece(*action)

        reward = 0
        winner = self.board.check_win()
        if winner != -1:
            reward = (-1)**(1-winner)

        terminated = winner != -1
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
    entry_point="environment:ChineseCheckersEnv",
)