import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()


class Direction(Enum):
    # Enumerate class for each snake direction
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')
# defining a named tuple class

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
# defining RGB color values

BLOCK_SIZE = 20
SPEED = 40
# game speed, modify to slow down/speed up the game


class SnakeGame:

    """
    Initiates PyGame, setting class variables and beginning the game by calling reset_game().
    """
    def __init__(self, width=800, height=600):
        pygame.display.set_caption('Snake')
        self.width = width
        self.height = height
        self.direction = Direction.RIGHT
        self.move_iteration = None
        self.apple = None
        self.score = None
        self.snake = None
        self.head = None
        self.BLOCK_SIZE = 20
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset_game()

    '''
    Resets game state after each win/loss by setting snake location, 
    length, score, and placing an apple.
    '''
    def reset_game(self):

        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.apple = None
        self.place_apple()
        self.move_iteration = 0
        # counter variable to track number of movements per game

    '''
    Generates random (x,y) coordinate that is not occupied by the snake and places an apple.
    '''
    def place_apple(self):
        x_coord = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y_coord = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.apple = Point(x_coord, y_coord)
        if self.apple in self.snake:
            self.place_apple()

    '''
    At each AI step, the game will move the snake in some direction and checks
    if it has collided and sets a reward indicating whether the snake has 
    died/taken too long (-10) or found an apple (+10).
    '''
    def play_step(self, action):
        self.move_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.collided() or self.move_iteration > 100 * len(self.snake):
            # died or took too long to find an apple
            reward = -10
            game_over = True
            return reward, game_over, self.score

        if self.head == self.apple:
            # found an apple
            self.score += 1
            reward = 10
            self.place_apple()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED * 20)
        return reward, game_over, self.score

    '''
    Checks if the snake head is out of bounds or hitting its tail.
    '''
    def collided(self, point=None):
        if point is None:
            point = self.head
        if point.x > (self.width - self.BLOCK_SIZE) or point.x < 0 or point.y > (self.height - self.BLOCK_SIZE) or point.y < 0:
            return True
        if point in self.snake[1:]:
            return True

        return False

    '''
    Paints the game by drawing rectangles to display the snake and text for the current score.
    '''
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.apple.x, self.apple.y, BLOCK_SIZE, BLOCK_SIZE))

        text = pygame.font.Font(size=50).render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    '''
    Method that indicates one movement (Left, Right, or Straight). Takes in an action array and converts it 
    to a direction based on the current direction.
    [1, 0, 0] --> straight
    [0, 1, 0] --> 90 deg. left
    [0, 0, 1] --> 90 deg. right
    '''
    def move(self, action):
        direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = direction.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            nextDirection = direction[index]
        elif np.array_equal(action, [0, 1, 0]):
            nextDirection = direction[(index + 1) % 4]
        else:
            nextDirection = direction[(index - 1) % 4]

        self.direction = nextDirection

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)
