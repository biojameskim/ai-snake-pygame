import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 100
LR = 0.001 # learning rate

class Agent:
    
    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # controls the randomness
        self.gamma = 0.9 # discount rate
        # If we exceed the maxlen, then deque will automatically remove elements from the left --> popleft()
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        
        head = game.snake[0]
        # Get the border positions of the head block
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Booleans to check which direction the snake is moving.
        # Only one of them should be true.
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State array
        state = [
            # There is a danger of collision straight ahead
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # There is a danger of collision if the snake turns right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # There is a danger of collision if the snake turns left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # The direction the snake is moving (only one of them is True)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # The location of the food
            game.food.x < game.head.x, # food is to the left of the snake
            game.food.x > game.head.x, # food is to the right of the snake
            game.food.y < game.head.y, # food is above the snake
            game.food.y > game.head.y # food is below the snake
        ]

        # Return state array and convert all Booleans to Int (0 or 1)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        # Append all information in the deque
        self.memory.append(
            (state, action, reward, next_state, game_over) # Append as a tuple
        ) # will automatically popleft if MAX_MEMORY is reached
    
    """
    Why do we need both train_long_memory and train_short_memory?
    train_long_memory is used to train the model based on the memory we have collected so far.
    train_short_memory is used to train the model based on the current game state.
    We need both because we want to train the model based on the current game state, but we also want to train the model based on the memory we have collected so far.
    We want to train the model based on the memory we have collected so far because we want to train the model based on the previous game states.
    We want to train the model based on the previous game states because we want the snake to learn from its mistakes.
    For example, if the snake dies, we want the snake to learn that it should not have made the move that caused it to die.
    However, we also want to train the model based on the current game state because we want the snake to learn from the current game state.
    For example, if the snake is about to die, we want the snake to learn that it should not make the move that will cause it to die.
    """

    def train_long_memory(self):
        # If our memory exceeds the BATCH_SIZE
        if len(self.memory) > BATCH_SIZE:
            # mini_sample randomly samples BATCH_SIZE number of tuples from self.memory
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory

        # zip(*mini_sample) returns one giant tuple with 5 tuples inside.
        # The first tuple contains all the states from all tuplies in mini_sample,
        # the second tuple contains all the actions from all tuplies in mini_sample, and so on.
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        """
        Trains for one game step
        """
        self.trainer.train_step(state, action, reward, next_state, game_over)
    
    def get_action(self, state):
        """
        Get next action based on the state
        """
        # Here, there is a tradeoff between exploration / exploitation

        # At the start of the training, we want to explore the environment so the snake will make a random move.
        # However, as we start training the model, we don't want the snake to keep making random moves.
        # Thus, we only make a random move if random.randint(0, 200) < self.epsilon, and the more games we play (self.n_games increases),
        # the less likely random.randint(0, 200) < self.epsilon will be true since epsilon keeps getting smaller.
        self.epsilon = 80 - self.n_games
        next_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
            next_move[idx] = 1

        # Else, we do a move based on our trained model
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # pytorch will execute the forward function defined in model.py
            idx = torch.argmax(prediction).item()
            next_move[idx] = 1

        return next_move
    
def train():
    plot_scores = [] # Keeps track of the scores
    plot_mean_scores = [] # Keeps track of the average scores
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get current state
        state_curr = agent.get_state(game)

        # get move based on current state
        next_move = agent.get_action(state_curr)

        # perform move and get new state
        reward, game_over, score = game.play_step(next_move)
        state_new = agent.get_state(game)

        # train short memory (only for 1 play_step)
        agent.train_short_memory(state_curr, next_move, reward, state_new, game_over)

        # remember (and store in memory)
        agent.remember(state_curr, next_move, reward, state_new, game_over)

        if game_over:
            game.reset() # reset game
            agent.n_games += 1 # increment number of games agent has played
            agent.train_long_memory() # train long memory

            if score > best_score: 
                best_score = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()