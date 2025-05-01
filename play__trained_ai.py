import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, device

# Load model
model = Linear_QNet(2308, 256, 3).to(device)
model.load()

# Setup game
game = SnakeGameAI(render=True)

# State building (same as in your training)
def get_state(game):
    head = game.snake[0]
    board_width = game.w // 20
    board_height = game.h // 20
    board_vision = np.zeros((board_height, board_width, 3), dtype=int)

    for i in range(board_height):
        for j in range(board_width):
            x = j * 20
            y = i * 20
            point = Point(x, y)
            board_vision[i, j, 0] = 1 if game.is_collision(point) else 0
            board_vision[i, j, 1] = 1 if point == game.food else 0
            board_vision[i, j, 2] = 1 if point == head else 0

    vision_state = board_vision.flatten()
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = np.concatenate([vision_state, [dir_l, dir_r, dir_u, dir_d]])
    return state

# Play one full game
while True:
    state = get_state(game)
    state_tensor = torch.tensor(state, dtype=torch.float).to(device)
    prediction = model(state_tensor)
    move = torch.argmax(prediction).item()

    final_move = [0, 0, 0]
    final_move[move] = 1

    reward, done, score = game.play_step(final_move)

    if done:
        print(f"Final Score: {score}")
        break
