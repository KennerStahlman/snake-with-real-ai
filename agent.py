import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer, device  # Import device from model.py
from helper import plot
import pygame
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Memory and Training Parameters
MAX_MEMORY = 200_000  # Maximum number of experiences to store in memory
BATCH_SIZE = 4000     # Number of experiences to sample for training
LR = 0.001           # Learning rate - higher values learn faster but may be unstable

class Agent:
    def __init__(self, epsilon_start=1):
        self.n_games = 0
        self.epsilon = epsilon_start      # Start with more exploration
        self.gamma = 0.95      # Higher discount factor for better long-term planning
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay buffer
        self.model = Linear_QNet(11, 256, 3).to(device)    # Move model to GPU
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Try to load saved model
        model_path = './model/model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loaded saved model")
        else:
            print("No saved model found, starting fresh")
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State representation:
        # [danger_straight, danger_right, danger_left, 
        #  direction_left, direction_right, direction_up, direction_down,
        #  food_left, food_right, food_up, food_down]
        state = [
            # Danger straight
            (dir_l and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            ]       
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
            
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # More gradual epsilon decay

        final_move = [0,0,0]
        
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train(render_every, max_games):
    """
    Train the AI agent
    Parameters:
        render_every (int): Render every Nth game. Higher values = faster training
        max_games (int): Maximum number of games to train for
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    # Calculate target score based on screen size
    # Each block is 20x20 pixels, so divide screen dimensions by 20
    grid_width = game.w // 20
    grid_height = game.h // 20
    target_score = grid_width * grid_height  # Maximum possible score (filling entire grid)
    
    while agent.n_games < max_games:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        
        # Only render every Nth game
        should_render = agent.n_games % render_every == 0
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move, should_render)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
    
        if done:
            # train long memory, plot result
            if agent.n_games < 300:
                agent.epsilon *= 0.995
            elif agent.n_games < 1000:
                agent.epsilon *= 0.9975
            else:
                agent.epsilon = max(0.01, agent.epsilon * 0.999)

            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            # Save model every 200 games
            if agent.n_games % 200 == 0:
                agent.model.save()
                print(f'Saved model at game {agent.n_games}')

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Target:', target_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            print(agent.epsilon)
            
            # Early stopping if target score is reached
            if mean_score >= target_score:
                print(f"Training complete! Reached target score of {target_score} in {agent.n_games} games.")
                break
                
    print(f"Training finished after {agent.n_games} games. Final mean score: {mean_score}")

def play_with_trained_ai():
    """
    Run the game with the trained AI model
    """
    agent = Agent(epsilon_start=0)  # This will automatically load the saved model
    game = SnakeGameAI()
    
    while True:  # Run indefinitely
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move, True)  # Always render
        
        if done:
            print('Game Over! Score:', score)
            game.reset()

if __name__ == "__main__":
    # Uncomment one of these lines:
    
    # To train the AI:
    # train(render_every=50, max_games=100000)
    
    # To play with the trained AI:
    play_with_trained_ai()