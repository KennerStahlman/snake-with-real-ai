import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Game parameters
BLOCK_SIZE = 20      # Size of each grid block
SPEED = 40          # Game speed (higher = faster)

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        
        # Fixed grid dimensions
        self.grid_width = 32  # Fixed number of tiles horizontally
        self.grid_height = 24  # Fixed number of tiles vertically
        
        # Calculate block size based on window dimensions
        self.block_size_x = self.w / self.grid_width
        self.block_size_y = self.h / self.grid_height
        
        self.reset()
        self.target_score = self.grid_width * self.grid_height
        self.max_moves_without_food = 3 * self.target_score

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        # Start in the middle of the grid
        grid_x = self.grid_width // 2
        grid_y = self.grid_height // 2
        self.head = Point(grid_x * self.block_size_x, grid_y * self.block_size_y)
        self.snake = [self.head,
                      Point((grid_x-1) * self.block_size_x, grid_y * self.block_size_y),
                      Point((grid_x-2) * self.block_size_x, grid_y * self.block_size_y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.moves_since_last_food = 0

    def _place_food(self):
        # Place food at random grid position
        grid_x = random.randint(0, self.grid_width - 1)
        grid_y = random.randint(0, self.grid_height - 1)
        self.food = Point(grid_x * self.block_size_x, grid_y * self.block_size_y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        self.moves_since_last_food += 1
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        
        # Check for timeout
        if self.moves_since_last_food > self.max_moves_without_food:
            game_over = True
            reward = -50
            return reward, game_over, self.score
            
        if self.is_collision():
            game_over = True
            reward = -50
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 100
            self._place_food()
            self.moves_since_last_food = 0  # Reset the counter when food is eaten
        else:
            self.snake.pop()
            reward = -0.1  # Small negative reward for each step
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Convert to grid coordinates for boundary check
        grid_x = round(pt.x / self.block_size_x)
        grid_y = round(pt.y / self.block_size_y)
        
        # hits boundary
        if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        # [straight, right, left]

        # Clockwise directions for turning
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Determine new direction based on action
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        # Move head in new direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size_x
        elif self.direction == Direction.LEFT:
            x -= self.block_size_x
        elif self.direction == Direction.DOWN:
            y += self.block_size_y
        elif self.direction == Direction.UP:
            y -= self.block_size_y

        self.head = Point(x, y) 