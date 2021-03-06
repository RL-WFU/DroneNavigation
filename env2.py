import random
import sys
from ICRSsimulator import *
import numpy as np
from copy import deepcopy

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
        self.local_map_lower_row = 0
        self.local_map_lower_col = 0
        self.local_map_upper_row = 24
        self.local_map_upper_col = 24

        # Define partitions
        self.region_one = [0, 0]
        self.region_two = [0, 60]
        self.region_three = [0, 120]
        self.region_four = [60, 0]
        self.region_five = [60, 60]
        self.region_six = [60, 120]
        self.region_seven = [120, 0]
        self.region_eight = [120, 60]
        self.region_nine = [120, 120]

        # Define initial targets
        self.targets = []
        self.targets.append([self.region_one[0] + self.totalRows/6, self.region_one[1] + self.totalCols/6])
        self.targets.append([self.region_two[0] + self.totalRows/6, self.region_two[1] + self.totalCols/6])
        self.targets.append([self.region_three[0] + self.totalRows/6, self.region_three[1] + self.totalCols/6])
        self.targets.append([self.region_four[0] + self.totalRows/6, self.region_four[1] + self.totalCols/6])
        self.targets.append([self.region_five[0] + self.totalRows/6, self.region_five[1] + self.totalCols/6])
        self.targets.append([self.region_six[0] + self.totalRows/6, self.region_six[1] + self.totalCols/6])
        self.targets.append([self.region_seven[0] + self.totalRows/6, self.region_seven[1] + self.totalCols/6])
        self.targets.append([self.region_eight[0] + self.totalRows/6, self.region_eight[1] + self.totalCols/6])
        self.targets.append([self.region_nine[0] + self.totalRows / 6, self.region_nine[1] + self.totalCols / 6])
        self.local_target = [24, 24]  # TODO: this is just if you start in the top right-hand corner of the region
        self.current_target_index = 0
        self.current_target = self.targets[self.current_target_index]

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
        self.TARGET_REWARD = 700
        self.END_REWARD = 2000
        self.VISITED_PENALTY = -5
        self.HOVER_PENALTY = -10
        self.INVALID_PENALTY = -20  # TODO: do we need this or should we hard code to stay in bounds

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

        # Reset initial targets
        self.targets = []
        self.targets.append([self.region_one[0] + self.totalRows/6, self.region_one[1] + self.totalCols/6])
        self.targets.append([self.region_two[0] + self.totalRows/6, self.region_two[1] + self.totalCols/6])
        self.targets.append([self.region_three[0] + self.totalRows/6, self.region_three[1] + self.totalCols/6])
        self.targets.append([self.region_six[0] + self.totalRows/6, self.region_six[1] + self.totalCols/6])
        self.targets.append([self.region_five[0] + self.totalRows / 6, self.region_five[1] + self.totalCols / 6])
        self.targets.append([self.region_four[0] + self.totalRows/6, self.region_four[1] + self.totalCols/6])
        self.targets.append([self.region_seven[0] + self.totalRows/6, self.region_seven[1] + self.totalCols/6])
        self.targets.append([self.region_eight[0] + self.totalRows/6, self.region_eight[1] + self.totalCols/6])
        self.targets.append([self.region_nine[0] + self.totalRows / 6, self.region_nine[1] + self.totalCols / 6])
        self.local_target = [24, 24]  # TODO: this is just if you start in the top right-hand corner of the region
        self.current_target_index = 0
        self.current_target = self.targets[self.current_target_index]

        # Reset env parameters
        self.start_row = random.randint(12, 42)
        self.start_col = random.randint(12, 42)
        self.row_position = self.start_row
        self.col_position = self.start_col

        # Get new drone state
        state = self.get_classified_drone_image()
        state = self.flatten_state(state)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, self.local_target[0] - self.row_position)
        state = np.append(state, self.local_target[1] - self.col_position)
        state = np.reshape(state, [1, 1, self.vision_size + 6])

        # Get new local map
        self.next_local_map()
        self.local_map = self.get_local_map()
        flattened_local_map = self.local_map.reshape(1, 1, 625)

        return state, flattened_local_map

    def step(self, action, time):
        self.done = False
        next_row = self.row_position
        next_col = self.col_position

        # Drone not allowed to move outside of the current local map (updates when target is reached)
        if action == 0:
            if self.row_position < self.local_map_upper_row and self.row_position < (
                    self.totalRows - self.sight_distance - 1):  # Forward one grid
                next_row = self.row_position + 1
                next_col = self.col_position
            else:
                action = 5
        elif action == 1:
            if self.col_position < self.local_map_upper_col and self.col_position < (
                    self.totalCols - self.sight_distance - 1):  # right one grid
                next_row = self.row_position
                next_col = self.col_position + 1
            else:
                action = 5
        elif action == 2:
            if self.row_position > self.local_map_lower_row and self.row_position > self.sight_distance + 1:  # back one grid
                next_row = self.row_position - 1
                next_col = self.col_position
            else:
                action = 5
        elif action == 3:
            if self.col_position > self.local_map_lower_col and self.col_position > self.sight_distance + 1:  # left one grid
                next_row = self.row_position
                next_col = self.col_position - 1
            else:
                action = 5
        if action == 5:  # This hardcodes the drone to move towards the target if it tries to take an invalid action
            if self.row_position < self.local_map_upper_row and self.row_position < (
                    self.totalRows - self.sight_distance - 1) and self.row_position < self.local_target[0]:
                next_row = self.row_position + 1
                next_col = self.col_position
            elif self.col_position < self.local_map_upper_col and self.col_position < (
                    self.totalCols - self.sight_distance - 1) and self.col_position < self.local_target[0]:
                next_row = self.row_position
                next_col = self.col_position + 1
            elif self.row_position > self.local_map_lower_row and self.row_position > self.sight_distance + 1 and self.row_position > \
                    self.local_target[0]:
                next_row = self.row_position - 1
                next_col = self.col_position
            elif self.col_position > self.local_map_lower_col and self.col_position > self.sight_distance + 1 and self.col_position > \
                    self.local_target[1]:
                next_row = self.row_position
                next_col = self.col_position - 1

        self.row_position = next_row
        self.col_position = next_col
        # print(self.row_position, self.col_position)
        # print(self.local_target)

        image = self.get_classified_drone_image()

        reward = self.get_reward(image, action)

        self.visited_position()
        self.update_map(image)

        # TODO: can instead set the done condition to be target reached
        if time > self.config.max_steps or self.target_reached():
            '''
            if self.current_target == 2:
                self.done = True
            else:
                self.current_target += 1
                self.next_local_map()
                print("new target:", self.current_target)
            '''
            self.done = True

        # TODO: include this once the end goal isn't the local target
        if self.local_target_reached():
            self.next_local_map()

        self.local_map = self.get_local_map()
        flattened_local_map = self.local_map.reshape(1, 1, 625)

        state = self.flatten_state(image)
        state = np.append(state, self.visited[self.row_position + 1, self.col_position])
        state = np.append(state, self.visited[self.row_position, self.col_position + 1])
        state = np.append(state, self.visited[self.row_position - 1, self.col_position])
        state = np.append(state, self.visited[self.row_position, self.col_position + 1])
        state = np.append(state, self.local_target[0] - self.row_position)
        state = np.append(state, self.local_target[1] - self.col_position)
        state = np.reshape(state, [1, 1, self.vision_size + 6])

        return state, flattened_local_map, reward, self.done

    def get_reward(self, image, action):
        """
        Calculates reward based on target, mining seen, and whether the current state has already been visited
        :param image: 2d array of mining probabilities within the drone's vision
        :return: reward value
        """
        mining_prob = 2*image[self.sight_distance, self.sight_distance]

        reward = mining_prob*self.MINING_REWARD*self.visited[self.row_position, self.col_position]
        reward += self.local_target_reached()*self.TARGET_REWARD + self.target_reached()*self.END_REWARD

        if action == 4 or action == 5:
            reward += self.HOVER_PENALTY

        if self.visited[self.row_position, self.col_position] == 0:
            reward += self.VISITED_PENALTY
        return reward

    def get_local_map(self):
        """
        Creates local map (shape: 25x25) of mining areas from the region map
        TODO: should this incorporate visited?
        :return: local_map
        """
        #local_map = np.zeros([25, 25])
        local_map = deepcopy(self.map[self.local_map_lower_row:self.local_map_upper_row+1, self.local_map_lower_col:self.local_map_upper_col+1])
        #row, col = self.get_local_target(self.target_one) # TODO: based on which target actually aiming for
        local_map[(self.local_target[0]-self.local_map_lower_row), (self.local_target[1]-self.local_map_lower_col)] = 1
        return local_map

    def next_local_map(self):
        """
        Sets boundaries on local map, placing the drone at the center of the new map
        :return: void
        """
        self.local_map_lower_row = self.row_position - 12
        self.local_map_upper_row = self.row_position + 12
        self.local_map_lower_col = self.col_position - 12
        self.local_map_upper_col = self.col_position + 12

        self.get_local_target()
        # print('next local map')

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
                self.map[self.row_position + i - self.sight_distance, self.col_position + j - self.sight_distance] = image[i, j]*2

    def target_reached(self):
        """
        :return: boolean, true if drone is at the target position
        """
        target = False
        if self.row_position == self.current_target[0] and self.col_position == self.current_target[1]:
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

    def get_next_target(self):
        # TODO: implement function (network??) to pick the next target
        self.current_target_index += 1
        self.current_target = self.targets[self.current_target_index]
        self.next_local_map()
        self.local_map = self.get_local_map()
        return self.current_target

    def get_local_target(self):
        """
        Sets the local map target based on where the drone is located in relation to the main target
        :param target: current target position (x, y)
        :return: row and col of the local map target
        """
        target = self.current_target
        row = 0
        col = 0
        #print('target:', target)
        if self.row_position == target[0]:
            row = self.row_position - self.local_map_lower_row
        elif self.row_position > target[0]:
            if self.row_position - target[0] > 12:
                row = 0
            else:
                row = target[0] - self.row_position + 12
        elif self.row_position < target[0]:
            if target[0] - self.row_position > 12:
                row = 24
            else:
                row = target[0] - self.row_position+ 12
        if self.col_position == target[1]:
            col = self.col_position - self.local_map_lower_col
        elif self.col_position > target[1]:
            if self.col_position - target[1] > 12:
                col = 0
            else:
                col = target[1] - self.col_position + 12
        elif self.col_position < target[1]:
            if target[1] - self.col_position > 12:
                col = 24
            else:
                col = target[1] - self.col_position + 12
        row = int(row+self.local_map_lower_row)
        col = int(col+self.local_map_lower_col)
        self.local_target = [row, col]

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

    def calculate_covered(self, size):
        covered = 0
        '''
               if size == 'local':
            for i in range(25):
                for j in range(25):
                    if self.visited[i][j] < 1:
                        covered += 1

            percent_covered = covered / (25*25)

        else:
            for i in range(self.totalRows):
                for j in range(self.totalCols):
                    if self.visited[i][j] < 1:
                        covered += 1

            percent_covered = covered / (self.totalCols * self.totalRows)

        '''
        for i in range(25):
            for j in range(25):
                if self.visited[i+self.local_map_lower_row][j+self.local_map_lower_col] < 1:
                    covered += 1

        percent_covered = covered / (25 * 25)

        return percent_covered

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
