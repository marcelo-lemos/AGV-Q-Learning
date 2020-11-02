import argparse
from argparse import Action
import logging
import numpy as np
import sys

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from random import Random

@dataclass(frozen=True)
class Point:

    x: int
    y: int

    def __add__(self, other):
        new_x = self.x + other.x
        new_y = self.y + other.y
        return Point(new_x, new_y)


class Actions(Enum):
    UP = Point(-1, 0)
    DOWN = Point(1, 0)
    RIGHT = Point(0, 1)
    LEFT = Point( 0, -1)


@dataclass(frozen=True)
class State:

    location: Point
    moves: int

    def update(self, grid, action):
        new_location = self.location + action.value
        if new_location.x < -1 or new_location.x > grid.shape[0]-1:
            new_location = self.location

        if new_location.y < -1 or new_location.y > grid.shape[1]-1:
            new_location = self.location

        if grid[new_location.x][new_location.y] == '*':
            new_location = self.location

        if grid[new_location.x][new_location.y] == '#':
            new_moves = 0
        else:
            new_moves = self.moves + 1

        return State(new_location, new_moves)


class QModel:
    
    __slots__ = ['logger', 'alpha', 'gamma', 'qvalues']

    def __init__(self, alpha, gamma):
        self.logger = logging.getLogger('Q-model')
        self.logger.info('Initializing Q-model')

        self.alpha = alpha
        self.logger.debug(f'Alpha set to: {alpha}')

        self.gamma = gamma
        self.logger.debug(f'Gamma set to: {gamma}')

        self.qvalues = defaultdict(lambda: {Actions.UP: 0.0, Actions.DOWN: 0.0, Actions.RIGHT: 0.0, Actions.LEFT: 0.0})

        self.logger.info('Q-model initialized')

    def get_q_value(self, state, action):
        return self.qvalues[state][action]
        pass

    def get_best_action(self, state):
        return max(self.qvalues[state], key=self.qvalues[state].get)
        pass

    def update_q_value(self, state, action, reward, next_state):
        self.logger.info('Q-Value update started')

        self.logger.debug(f'S: {state}')
        self.logger.debug(f'A: {action}')
        self.logger.debug(f'R: {reward}')
        self.logger.debug(f"S': {next_state}")

        self.logger.debug(f'Old Q-value: {self.qvalues[state][action]}')

        # Get next state best Q-value
        next_state_action = self.get_best_action(next_state)
        next_state_qvalue = self.get_q_value(next_state, next_state_action)
        self.logger.debug(f'Next state best Q-value: {next_state_qvalue}')

        # Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max(Q(s',a)) - Q(s,a))
        self.qvalues[state][action] += self.alpha * (reward + self.gamma * next_state_qvalue - self.qvalues[state][action])
        self.logger.info('Q-Value updated')
        self.logger.debug(f'New Q-value: {self.qvalues[state][action]}')


class EpsilonGreedyStrategy:

    __slots__ = ['logger', 'epsilon', 'random']

    def __init__(self, epsilon, seed=None):
        self.logger = logging.getLogger('Epsilon Greedy Strategy')
        self.logger.info('Initializing Epsilon Greedy Strategy')

        self.epsilon = epsilon
        self.logger.debug(f'Epsilon set to: {epsilon}')

        if seed is None:
            self.random = Random(datetime.now())
            self.logger.debug('No seed specified. Seed set to current time')
        else:
            self.random = Random(seed)
            self.logger.debug(f'Seed set to: {seed}')

        self.logger.info('Epsilon Greedy Strategy initialized')

    def choose_action(self, state, model):
        self.logger.info('Choosing action')
        random_number = self.random.random()
        self.logger.debug(f'Random number generated: {random_number}')

        action = None

        if(random_number < self.epsilon):
            self.logger.info('Exploring')
            action = self.random.choice(list(Actions))
        else:
            self.logger.info('Exploiting')
            action = model.get_best_action(state)

        self.logger.debug(f'Action chosen: {action}')
        return action


class Learning_Agent:

    __slots__ = ['logger', 'random', 'grid', 'max_distance', 'state', 'model', 'strategy']

    def __init__(self, grid, max_distance, model, strategy, seed=None):
        self.logger = logging.getLogger('Learning Agent')
        self.logger.info('Initializing Learning Agent')

        self.max_distance = max_distance
        self.logger.debug(f'Max distance set to: {max_distance}')

        if seed is None:
            self.random = Random(datetime.now())
            self.logger.debug('No seed specified. Seed set to current time')
        else:
            self.random = Random(seed)
            self.logger.debug(f'Seed set to: {seed}')

        self.grid = grid
        self.model = model
        self.strategy = strategy

        self.logger.info('Learning Agent initialized')

    def reset(self):
        self.logger.info('Reseting agent to a random location')
        x = self.random.randrange(self.grid.shape[0])
        y = self.random.randrange(self.grid.shape[1])

        while self.grid[x][y] == '*' or self.grid[x][y] == '$':
            x = self.random.randrange(self.grid.shape[0])
            y = self.random.randrange(self.grid.shape[1])

        location = Point(x, y)
        self.logger.debug(f'New location: {location}')

        self.state = State(location, 0)
        self.logger.info('Agent reset')

    def get_reward(self, action, next_state):
        rewards = {
            'wall': -10,
            'lost': -10,
            'checkpoint': 1,
            'goal': 10,
            'default': -1
        }

        if self.state.location == next_state.location:
            self.logger.debug(f"Hit a wall. Reward: {rewards['wall']}")
            return rewards['wall']

        if next_state.moves > self.max_distance:
            self.logger.debug(f"Got lost. Reward: {rewards['lost']}")
            return rewards['lost']

        if grid[next_state.location.x][next_state.location.y] == '#':
            self.logger.debug(f"Got to a checkpoint. Reward: {rewards['checkpoint']}")
            return rewards['checkpoint']

        if grid[next_state.location.x][next_state.location.y] == '$':
            self.logger.debug(f"Got to the goal. Reward: {rewards['goal']}")
            return rewards['goal']

        self.logger.debug(f"Default reward: {rewards['default']}")
        return rewards['default']

    def act(self):
        action = strategy.choose_action(self.state, self.model)
        next_state = self.state.update(self.grid, action)

        reward = self.get_reward(action, next_state)

        # Learn
        model.update_q_value(self.state, action, reward, next_state)

        self.state = next_state
        if self.state.moves > self.max_distance:
            # Episode over
            return False
        if grid[self.state.location.x][self.state.location.y] == '$':
            # Episode over
            return False

        return True


    def train(self, episodes):
        self.logger.info('Training started')

        for i in range(episodes):
            self.reset()
            self.logger.info(f'Episode {i} started')
            while self.act():
                pass
            self.logger.info(f'Episode {i} finished')

        self.logger.info('Training finished')


def load_grid(filename):
    logger = logging.getLogger('Grid')
    logger.info(f'Loading grid from file: {filename}')

    with open(filename) as f:
        content = f.read().splitlines()

    meta = content.pop(0).split()
    height, width, max_distance = [int(x) for x in meta]
    grid = np.array([list(row) for row in content])

    logger.info(f'Grid loaded')
    logger.debug(f'Grid:\n{grid}')

    return grid, int(meta[2])


if __name__ == '__main__':
    logging.basicConfig(
        filename='qlearning.log',
        level=logging.ERROR,
        format='%(asctime)s [%(levelname)8s] [%(name)s] %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S')

    logger = logging.getLogger('Main')

    logger.info('AGV started')

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, required=True)
    parser.add_argument('-g', '--gamma', type=float, required=True)
    parser.add_argument('-e', '--epsilon', type=float, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-n', '--episodes', type=int, required=True)
    cmd_args = parser.parse_args()

    grid, max_dist = load_grid(cmd_args.input)
    model = QModel(cmd_args.alpha, cmd_args.gamma)
    strategy = EpsilonGreedyStrategy(cmd_args.epsilon)
    agent = Learning_Agent(grid, max_dist, model, strategy)

    agent.train(cmd_args.episodes)

    action_dict = {
        Actions.UP: 'UP',
        Actions.DOWN: 'DOWN',
        Actions.RIGHT: 'RIGHT',
        Actions.LEFT: 'LEFT'
    }

    with open('pi.txt','w') as f:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(max_dist,-1,-1):
                    if grid[i][j] != '*' and grid[i][j] != '$':
                        state = State(Point(i, j), k)
                        action = model.get_best_action(state)
                        qvalue = model.get_q_value(state, action)
                        f.write(f'({state.location.x}, {state.location.y}, {max_dist-state.moves}), {action_dict[action]}, {qvalue:.2f}\n')

    logger.info('AGV finished')
