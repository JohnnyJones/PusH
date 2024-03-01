import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

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

    def __init__(self, render_mode=None, players=2):
        self.window_size = 512 # Size of the PyGame window

        self.observation_space = spaces.Dict(
            {

            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.COLORS["BACKGROUND"])
        
        
        