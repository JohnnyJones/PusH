import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class ChineseCheckersEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, players=2):
        self.window_size = 512 # Size of the PyGame window

        self.observation_space = spaces.Dict(
            {

            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        