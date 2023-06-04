import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        Resets the game
        """
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        # Snake always starts off with 3 blocks.
        # The head, and 2 body blocks that are relative to the head's position.
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        # Reset frame_iteration back to 0 (how many frames the snake has been moving)
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # if food is placed on the snake, redo placement
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        # Update the frame iteration for each play step
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False

        # Game should end when the snake collides or when the game is going on too long with no progress.
        # We define no progress as the snake moving 100 times without eating the food or colliding.
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        """
        If you look at the function _move, you'll see that the snake's head is always
        updated to a new position. After this, we always insert the head into the snake
        in our play_step loop (step 2). 

        Thus, if we didn't pop the last element of the snake for each iteration, 
        the snake would keep growing in length.
        
        Thus, we pop the last element of the snake if the snake's head is not on the food,
        and we don't pop the last element if the snake's head is on the food.
        """

        # If the snake's head is on the food, we don't pop the last element of the snake.
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # Pop the last element of the snake to give the illusion of "moving"
            # This is because we're inserting the new head at the beginning of the snake in step 2 (move)
            # If we don't pop the last element, the snake will grow in length even when it didn't eat the food
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        # This caps the fps of the game at SPEED.
        # If we didn't cap the fps then our while loop would run as fast as our computer will let it run and the game would be unplayable.
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        """
        Returns True if pt collides with a boundary or itself.
        """
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # action = [straight, right, left]
        # Only 1 of the 3 elements can be 1.

        # For example, if action = [1, 0, 0], the snake should keep moving straight (in current direction)
        #  if action = [0, 1, 0], then the snake should turn right (relative to its current direction)

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # idx is the index of the current direction of the snake
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change in direction
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4 # go clockwise based on current direction (r -> d -> l -> u)
            new_dir = clock_wise[new_idx] # Turn right 
        else: # np.array_equal(action, [0, 0, 1])
            new_idx = (idx - 1) % 4 # go counter-clockwise based on current direction (r -> u -> l -> d)
            new_dir = clock_wise[new_idx] # Turn left

        self.direction = new_dir

        # Get (x, y) of current head
        x = self.head.x
        y = self.head.y

        # Update (x, y) of current head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE # y-axis starts from 0 at the top and increases as you go down
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            