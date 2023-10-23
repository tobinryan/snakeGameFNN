import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import Linear_Net, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Trainer:
    """
    Initiates the training module for the FNN and instantiates the memory as a deque, the model used to train the FNN,
    and the trainer using defined learning rates and gamma value.
    """

    def __init__(self):
        self.num_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Net(11, 256, 3)
        self.trainer = QTrainer(model=self.model, learning_rate=LR, gamma=self.gamma)

    '''
    Takes in a current game and returns an array of the current game state:
    (danger (forward, right, left), directions, and relative apple location).
    '''
    @staticmethod
    def get_state(game):
        head = game.snake[0]
        point_l = Point(head.x - game.BLOCK_SIZE, head.y)
        point_r = Point(head.x + game.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - game.BLOCK_SIZE)
        point_d = Point(head.x, head.y + game.BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_l and game.collided(point_l)) or
            (dir_r and game.collided(point_r)) or
            (dir_u and game.collided(point_u)) or
            (dir_d and game.collided(point_d)),
            # checks if there is danger straight

            (dir_l and game.collided(point_u)) or
            (dir_r and game.collided(point_d)) or
            (dir_u and game.collided(point_r)) or
            (dir_d and game.collided(point_l)),
            # checks if there is danger right

            (dir_l and game.collided(point_d)) or
            (dir_r and game.collided(point_u)) or
            (dir_u and game.collided(point_l)) or
            (dir_d and game.collided(point_r)),
            # checks if there is danger left

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.apple.x < game.head.x,
            game.apple.x > game.head.x,
            game.apple.y < game.head.y,
            game.apple.y > game.head.y
            # checks relative location of the apple compared to the snake head
        ]

        return np.array(state, dtype=int)

    '''
    Adds the current game state and current rewards to the deque.
    '''
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    '''
    Trains on a large set of samples by calling train_step() on the set of samples.
    '''
    def train_large_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # if the memory is greater than our BATCH_SIZE, take a random sample of size BATCH_SIZE
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            # if the memory is not > BATCH_SIZE, use the entire memory as the sample
            sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    '''
    Trains on one sample, given states, action... 
    '''
    def train_small_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    '''
    Given a game state, predicts the next action to take and returns this as an array.
    '''
    def get_action(self, state):
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # allows for exploration by sometimes randomly choosing an action
            # (increases in probability as we reach 80 games)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # otherwise, converts the current state to a tensor and predicts a move
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            # chooses the highest probability prediction
            final_move[move] = 1

        return final_move


'''
Trains the FNN by repeatedly predicting and playing actions based on the current game state, while simultaneously
saving this to memory.
'''
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    trainer = Trainer()
    game = SnakeGame()
    while True:
        current_state = trainer.get_state(game)
        move = trainer.get_action(current_state)
        reward, game_over, score = game.play_step(move)
        new_state = trainer.get_state(game)
        # gets the current game state, predicts a move, playing that move, and calculating the new game state

        trainer.train_small_memory(current_state, move, reward, new_state, game_over)
        trainer.remember(current_state, move, reward, new_state, game_over)
        # trains the memory and saves the states to the deque

        if game_over:
            # when the game ends, resets the game and trains the memory
            game.reset_game()
            trainer.num_games += 1
            trainer.train_large_memory()

            if score >= best_score:
                best_score = score
                trainer.model.save()

            print('Game', trainer.num_games, 'Score', score, 'Record', best_score)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / trainer.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
