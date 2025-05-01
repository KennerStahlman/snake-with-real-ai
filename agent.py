import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer, device
from helper import plot
import os
from torch.cuda.amp import autocast, GradScaler

MAX_MEMORY = 1_000_000
BATCH_SIZE = 32768
LR = 0.001

class Agent:
    def __init__(self, epsilon_start=1):
        self.n_games = 0
        self.epsilon = epsilon_start
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(2304, 512, 256, 128, 3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.scaler = GradScaler()

        model_path = './model/model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loaded saved model")
        else:
            print("No saved model found, starting fresh")

    def get_state(self, game):
        state = np.zeros((game.grid_width, game.grid_height, 3), dtype=np.float32)
        for point in game.snake[1:]:
            grid_x = int(point.x / game.block_size_x)
            grid_y = int(point.y / game.block_size_y)
            if 0 <= grid_x < game.grid_width and 0 <= grid_y < game.grid_height:
                state[grid_x, grid_y, 1] = 1
        head_x = int(game.head.x / game.block_size_x)
        head_y = int(game.head.y / game.block_size_y)
        if 0 <= head_x < game.grid_width and 0 <= head_y < game.grid_height:
            state[head_x, head_y, 0] = 1
        food_x = int(game.food.x / game.block_size_x)
        food_y = int(game.food.y / game.block_size_y)
        if 0 <= food_x < game.grid_width and 0 <= food_y < game.grid_height:
            state[food_x, food_y, 2] = 1
        return state.flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.scaler)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done, self.scaler)

    def get_action(self, state):
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            with autocast():
                prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train(render_every, max_games):
    if os.path.exists('./model/model.pth'):
        os.remove('./model/model.pth')
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    grid_width = game.w // 20
    grid_height = game.h // 20
    target_score = grid_width * grid_height

    while agent.n_games < max_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            agent.epsilon = max(0.001, agent.epsilon * 0.999)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            if agent.n_games % 10 == 0:
                plot(plot_scores, plot_mean_scores)
            if agent.n_games % 200 == 0:
                agent.model.save()
                print(f'Saved model at game {agent.n_games}')
            if mean_score >= target_score:
                print(f"Training complete! Reached target score of {target_score} in {agent.n_games} games.")
                break
    print(f"Training finished after {agent.n_games} games. Final mean score: {mean_score}")

def play_with_trained_ai():
    agent = Agent(epsilon_start=0)
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        if done:
            print('Game Over! Score:', score)
            game.reset()
