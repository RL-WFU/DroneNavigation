import random
import sys
from ICRSsimulator import *
import numpy as np

class Env:
    def __init__(self, config):
        # Create simulator object and load the image
        self.config = config
        self.sim = ICRSsimulator(config.image)
        if not self.sim.loadImage():
            print("Error: could not load image")
            sys.exit(0)

        # Initialize map of size totalRows x totalCols from the loaded image
        self.totalRows = config.total_rows
        self.totalCols = config.total_cols
        self.set_simulation_map()

        # Initialize tracking
        self.map = np.zeros([self.totalRows, self.totalCols])
        self.visited = np.ones([self.totalRows, self.totalCols])
        self.local_map = np.zeros([25, 25])
        self.local_map_lower = 0
        self.local_map_upper = 24

        # Define partitions
        self.region_one = [0, 0]
        self.region_two = [0, self.totalCols/2]
        self.region_three = [self.totalRows/2, 0]
        self.region_four = [self.totalRows/2, self.totalCols/2]

        # Define initial targets
        self.target_one = [self.region_one[0] + self.totalRows/4, self.region_one[1] + self.totalCols/4]
        self.target_two = [self.region_two[0] + self.totalRows/4, self.region_two[1] + self.totalCols/4]
        self.target_three = [self.region_three[0] + self.totalRows/4, self.region_three[1] + self.totalCols/4]
        self.target_four = [self.region_four[0] + self.totalRows/4, self.region_four[1] + self.totalCols/4]
        self.local_target = [24, 24] # TODO: this is just if you start in the top right-hand corner of the region

        # Set env parameters
        self.num_actions = 5
        self.sight_distance = 2
        self.vision_size = (self.sight_distance * 2 + 1) * (self.sight_distance * 2 + 1)
        self.start_row = self.sight_distance
        self.start_col = self.sight_distance
        self.row_position = self.start_row
        self.col_position = self.start_col

        # Set reward values
        self.MINING_REWARD = 50
        self.TARGET_REWARD = 200
        self.VISITED_PENALTY = -5
        self.INVALID_PENALTY = -20 # TODO: do we need this or should we hard code to stay in bounds

    def reset_environment(self):
        """
        Resets the environment at the beginning of a new episode
        :return: state of the drone: flattened array of probabilities of mining in each cell within its vision with
                 the binary "visited" tag for each of the four next positions
                 TODO: consider appending the target position and distance (is local map alone enough?)
        """
        # Initialize tracking
        self.map = np.zeros([self.totalRows, self.totalCols])
        self.visited = np.ones([self.totalRows, self.totalCols])
        self.local_map = np.zeros([self.totalRows / 8, self.totalCols / 8])

        # Reset partitions
        self.region_one = [0, 0]
        self.region_two = [0, self.totalCols / 2]
        self.region_three = [self.totalRows / 2, 0]
        self.region_four = [self.totalRows / 2, self.totalCols / 2]

        # Reset initial targets
        self.target_one = self.region_one + self.totalRows / 4
        self.target_two = self.region_two + self.totalRows / 4
        self.target_three = self.region_three + self.totalRows / 4
        self.target_four = self.region_four + self.totalRows / 4

        # Reset env parameters
        self.start_row = self.sight_distance
        self.start_col = self.sight_distance
        self.row_position = self.start_row
        self.col_position = self.start_col

        # Get new drone state
        state = self.get_classified_drone_image()
        state = self.flatten_state(state)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.reshape(state, [1, self.vision_size + 4])

        return state

    def step(self, action, time):
        self.done = False

        next_row = self.row_position
        next_col = self.col_position

        # Drone not allowed to move outside of the current local map (updates when target is reached)
        if action == 0:
            if self.row_position < self.local_map_upper:  # Forward one grid
                next_row = self.row_position + 1
                next_col = self.col_position
            else:
                action = 4
        elif action == 1:
            if self.col_position < self.local_map_upper:  # right one grid
                next_row = self.row_position
                next_col = self.col_position + 1
            else:
                action = 4
        elif action == 2:
            if self.row_position > self.local_map_lower:  # back one grid
                next_row = self.row_position - 1
                next_col = self.col_position
            else:
                action = 4
        elif action == 3:
            if self.col_position > self.local_map_lower:  # left one grid
                next_row = self.row_position
                next_col = self.col_position - 1
            else:
                action = 4

        self.row_position = next_row
        self.col_position = next_col

        image = self.get_classified_drone_image()
        state = self.flatten_state(image)
        state = np.append(state, self.visited[self.row_position + 1, self.col_position])
        state = np.append(state, self.visited[self.row_position, self.col_position + 1])
        state = np.append(state, self.visited[self.row_position - 1, self.col_position])
        state = np.append(state, self.visited[self.row_position, self.col_position + 1])
        state = np.reshape(state, [1, self.vision_size + 4])

        reward = self.get_reward(image)

        self.visited_position()
        self.update_map(image)

        if self.local_target_reached():
            self.next_local_map()
        self.local_map = self.get_local_map()

        # TODO: can instead set the done condition to be local target reached
        if time > self.config.max_steps or self.target_reached(1):
            self.done = True

        return state, self.local_map, reward, self.done

    def get_reward(self, image):
        """
        Calculates reward based on target, mining seen, and whether the current state has already been visited
        TODO: add reward for if the drone chooses an "invalid" action
        :param image: 2d array of mining probabilities within the drone's vision
        :return: reward value
        """
        mining_prob = image[self.sight_distance, self.sight_distance]

        reward = mining_prob*self.MINING_REWARD*self.visited[self.row_position, self.col_position] + \
            self.local_target_reached()*self.TARGET_REWARD + self.target_reached(self.target_one)*self.TARGET_REWARD

        if self.visited[self.row_position, self.col_position] == 0:
            reward += self.VISITED_PENALTY
        return reward

    def get_local_map(self):
        """
        Creates local map (shape: 25x25) of mining areas from the region map
        TODO: should this incorporate visited?
        :return: local_map
        """
        local_map = self.map[self.local_map_lower:self.local_map_upper, self.local_map_lower:self.local_map_upper]
        row, col = self.get_local_target(self.target_one) # TODO: based on which target actually aiming for
        local_map[row, col] = .5
        return local_map

    def next_local_map(self):
        """
        Sets boundaries on local map, placing the drone at the center of the new map
        :return: void
        """
        self.local_map_lower = self.row_position - 12
        self.local_map_upper = self.row_position + 12

    def visited_position(self):
        """
        Updates array that keeps track of the previous cells visited by the drone (doesn't consider peripheral vision)
        :return: void
        """
        self.visited[self.row_position, self.col_position] = 0

    def update_map(self, image):
        """
        Adds new state information to the map of the entire region
        :param image: 2d array of mining probabilities within the drone's vision
        :return: void
        """
        for i in range(self.sight_distance*2):
            for j in range(self.sight_distance*2):
                self.map[self.row_position + i - self.sight_distance, self.col_position + j - self.sight_distance] = image[i, j]

    def target_reached(self, region):
        """
        :return: boolean, true if drone is at the target position
        """
        target = False
        if self.row_position == self.target_one[0] and self.col_position == self.target_one[1] and region == 1:
            target = True
        elif self.row_position == self.target_two[0] and self.col_position == self.target_two[1] and region == 2:
            target = True
        elif self.row_position == self.target_three[0] and self.col_position == self.target_three[1] and region == 3:
            target = True
        elif self.row_position == self.target_four[0] and self.col_position == self.target_four[1] and region == 4:
            target = True
        return target

    def local_target_reached(self):
        """
        :return: boolean, true if drone is at the local target position
        """
        target = False
        if self.row_position == self.local_target[0] and self.col_position == self.local_target[1]:
            target = True
        return target

    def get_local_target(self, target):
        """
        Sets the local map target based on where the drone is located in relation to the main target
        :param target: current target position (x, y)
        :return: row and col of the local map target
        """
        row = 0
        col = 0
        if self.row_position == target[0]:
            row = self.row_position
        elif self.row_position > target[0]:
            row = 0
        elif self.row_position < target[0]:
            row = 24
        if self.col_position == target[1]:
            col = self.col_position
        elif self.col_position > target[1]:
            col = 0
        elif self.col_position < target[1]:
            col = 24
        self.local_target = [row+self.local_map_upper, col+self.local_map_upper]
        return row, col

    def get_classified_drone_image(self):
        """
        :return: mining probability for each cell within the drone's current vision, flattened
        """
        self.sim.setDroneImgSize(self.sight_distance, self.sight_distance)
        self.sim.setNavigationMap()
        image = self.sim.getClassifiedDroneImageAt(self.row_position, self.col_position)
        return image

    def save_local_map(self, fname):
        """
        Saves the current local map of the drone to an image file
        :param fname: name of image to save
        :return: void
        """
        plt.imshow(self.local_map[:, :], cmap='gray', interpolation='none')
        plt.title("Local Map")
        plt.savefig(fname)
        # plt.show()
        plt.clf()

    def save_map(self, fname):
        """
        Saves the current region map of the drone to an image file
        :param fname: name of image to save
        :return: void
        """
        plt.imshow(self.map[:, :], cmap='gray', interpolation='none')
        plt.title("Region Map")
        plt.savefig(fname)
        plt.clf()

    def plot_path(self, fname):
        """
        Saves the current path of the drone to an image file
        :param fname: name of image to save
        :return: void
        """
        plt.imshow(self.visited[:, :], cmap='gray', interpolation='none')
        plt.title("Drone Path")
        plt.savefig(fname)
        plt.clf()

    def set_simulation_map(self):
        """
        sets the pixel value thresholds for ICRS classification
        :return: void
        """
        # Simulate classification of mining areas
        lower = np.array([80, 90, 70])
        upper = np.array([100, 115, 150])
        interest_value = 1  # Mark these areas as being of highest interest
        self.sim.classify('Mining', lower, upper, interest_value)

        # Simulate classification of forest areas
        lower = np.array([0, 49, 0])
        upper = np.array([80, 157, 138])
        interest_value = 0  # Mark these areas as being of no interest
        self.sim.classify('Forest', lower, upper, interest_value)

        # Simulate classification of water
        lower = np.array([92, 100, 90])
        upper = np.array([200, 190, 200])
        interest_value = 0  # Mark these areas as being of no interest
        self.sim.classify('Water', lower, upper, interest_value)

        self.sim.setMapSize(self.totalRows, self.totalCols)
        self.sim.createMap()

    def flatten_state(self, state):
        """
        :param state: 2d array of mining probabilities within the drone's vision
        :return: one dimensional array of the state information
        """
        flat_state = state.reshape(1, self.vision_size)
        return flat_state
