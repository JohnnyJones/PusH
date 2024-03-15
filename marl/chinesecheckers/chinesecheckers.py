import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import numpy as np
import pygame
import gymnasium as gym
import torch

from gymnasium import spaces
from gymnasium.envs.registration import register
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
        return str(self.board)
    
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
        return new_board

    def to_tensor(self) -> torch.Tensor:        
        if (n := len(self.history)) != 3:
            raise ValueError(f"History must have 3 states, got {n}")

        # concatenate history into single ndarray
        if self.turn == 0:
            # stack as is
            history = np.stack(self.history)
        else:
            # reverse the order of each state
            history = [state[::-1] for state in self.history]
            history = np.stack(history)
        
        # add layer for player turn
        history = np.concatenate((history, np.ones((1, 7, 7), dtype=np.int8) * self.turn))

        return torch.from_numpy(history)

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
            if all([self.board[i] == player_id for i in win_positions]):
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
        for row in self.board.board:
            print("[", end=" ")
            for cell in row:
                if cell == -1:
                    print(".", end=" ")
                else:
                    print(cell, end=" ")
            print("]")
        print()
    
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
        self.board.reset()

        observation = self._get_obs()
        info = self._get_info()

        self._render()
        
        return observation, info
    
    def step(self, action):
        piece_id, move_position = action
        move_position = Position(*move_position)
        valid_actions = self.board.get_action_mask(self.board.turn)
        if not valid_actions[piece_id, move_position.x, move_position.y]:
            raise ValueError("Invalid action")
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
    entry_point="chinesecheckers:ChineseCheckersEnv",
)