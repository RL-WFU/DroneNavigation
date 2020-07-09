from Environment.base_env import *
from copy import deepcopy


class Baseline(Env):
    def __init__(self):
        # Set simulation
        self.set_simulation_map()

        # Set task-specific parameters
        self.num_actions = 5

        # Set reward values
        self.MINING_REWARD = 100
        self.COVERAGE_REWARD = 200
        self.VISITED_PENALTY = -5
        self.HOVER_PENALTY = -10

        # Set starting position
        self.__class__.row_position = self.totalRows - self.sight_distance - 1
        self.__class__.col_position = self.totalCols - self.sight_distance - 1

    def step(self, action):
        next_row = self.__class__.row_position
        next_col = self.__class__.col_position

        # Drone not allowed to move outside of the current region
        if action == 0:  # Forward one grid
            if self.__class__.row_position < (self.totalRows - self.sight_distance - 1):
                next_row = self.__class__.row_position + 1
                next_col = self.__class__.col_position
            else:
                action = 4
        elif action == 1:  # right one grid
            if self.__class__.col_position < (self.totalCols - self.sight_distance - 1):
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position + 1
            else:
                action = 4
        elif action == 2:
            if self.__class__.row_position > self.sight_distance + 1:  # back one grid
                next_row = self.__class__.row_position - 1
                next_col = self.__class__.col_position
            else:
                action = 4
        elif action == 3:
            if self.__class__.col_position > self.sight_distance + 1:  # left one grid
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position - 1
            else:
                action = 4

        self.__class__.row_position = next_row
        self.__class__.col_position = next_col

        image = self.get_classified_drone_image()

        self.visited_position()
        self.update_map(image)