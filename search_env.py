from Environment.base_env import *


class Search(Env):
    def __init__(self):
        # Define initial targets
        self.local_target = [24, 24]
        self.__class__.current_target_index = 0
        self.current_target = self.targets[self.__class__.current_target_index]

        # Set simulation
        self.set_simulation_map()
        self.sim.showMap()

        # Set task-specific parameters
        self.num_actions = 5

        # Set reward values
        self.MINING_REWARD = 100
        self.TARGET_REWARD = 700
        self.END_REWARD = 100
        self.VISITED_PENALTY = -25
        self.HOVER_PENALTY = -25

    def reset_search(self, row, col):
        self.__class__.row_position = row
        self.__class__.col_position = col
        self.current_target = self.targets[self.__class__.current_target_index]
        print(self.current_target)

        # Get new local map
        self.next_local_map()
        self.local_map = self.get_local_map()
        flattened_local_map = self.local_map.reshape(1, 1, 625)

        # Get new drone state
        state = self.get_classified_drone_image()
        state = self.flatten_state(state)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, self.local_target[0] - self.__class__.row_position)
        state = np.append(state, self.local_target[1] - self.__class__.col_position)
        #state = np.append(state, self.calculate_covered('region'))
        state = np.reshape(state, [1, 1, self.vision_size + 6])

        return state, flattened_local_map

    def step(self, action, time):
        self.done = False
        next_row = self.__class__.row_position
        next_col = self.__class__.col_position

        # Drone not allowed to move outside of the current local map (updates when target is reached)
        if action == 0:
            if self.__class__.row_position < self.local_map_upper_row and self.__class__.row_position < (
                    self.totalRows - self.sight_distance - 1):  # Forward one grid
                next_row = self.__class__.row_position + 1
                next_col = self.__class__.col_position
            else:
                action = 5
        elif action == 1:
            if self.__class__.col_position < self.local_map_upper_col and self.__class__.col_position < (
                    self.totalCols - self.sight_distance - 1):  # right one grid
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position + 1
            else:
                action = 5
        elif action == 2:
            if self.__class__.row_position > self.local_map_lower_row and self.__class__.row_position > self.sight_distance + 1:  # back one grid
                next_row = self.__class__.row_position - 1
                next_col = self.__class__.col_position
            else:
                action = 5
        elif action == 3:
            if self.__class__.col_position > self.local_map_lower_col and self.__class__.col_position > self.sight_distance + 1:  # left one grid
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position - 1
            else:
                action = 5
        if action == 5 or action == 4:  # This hardcodes the drone to move towards the target if it tries to take an invalid action
            if self.__class__.row_position < self.local_map_upper_row and self.__class__.row_position < (
                    self.totalRows - self.sight_distance - 1) and self.__class__.row_position < self.local_target[0]:
                next_row = self.__class__.row_position + 1
                next_col = self.__class__.col_position
            elif self.__class__.col_position < self.local_map_upper_col and self.__class__.col_position < (
                    self.totalCols - self.sight_distance - 1) and self.__class__.col_position < self.local_target[1]:
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position + 1
            elif self.__class__.row_position > self.local_map_lower_row and self.__class__.row_position > \
                    self.sight_distance + 1 and self.__class__.row_position > self.local_target[0]:
                next_row = self.__class__.row_position - 1
                next_col = self.__class__.col_position
            elif self.__class__.col_position > self.local_map_lower_col and self.__class__.col_position > \
                    self.sight_distance + 1 and self.__class__.col_position > self.local_target[1]:
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position - 1

        self.__class__.row_position = next_row
        self.__class__.col_position = next_col
        #print(self.row_position, self.col_position)
        #print(self.local_target)

        image = self.get_classified_drone_image()

        reward = self.get_reward(image, action)

        self.visited_position()
        self.update_map(image)

        if time > self.config.max_steps_search or self.target_reached():
            self.done = True

        if self.local_target_reached():
            self.next_local_map()

        self.local_map = self.get_local_map()
        flattened_local_map = self.local_map.reshape(1, 1, 625)

        state = self.flatten_state(image)
        state = np.append(state, self.visited[self.__class__.row_position + 1, self.__class__.col_position])
        state = np.append(state, self.visited[self.__class__.row_position, self.__class__.col_position + 1])
        state = np.append(state, self.visited[self.__class__.row_position - 1, self.__class__.col_position])
        state = np.append(state, self.visited[self.__class__.row_position, self.__class__.col_position + 1])
        state = np.append(state, self.local_target[0] - self.__class__.row_position)
        state = np.append(state, self.local_target[1] - self.__class__.col_position)
        #state = np.append(state, self.calculate_covered('region'))
        state = np.reshape(state, [1, 1, self.vision_size + 6])

        return state, flattened_local_map, reward, self.done

    def get_reward(self, image, action):
        """
        Calculates reward based on target, mining seen, and whether the current state has already been visited
        :param image: 2d array of mining probabilities within the drone's vision
        :return: reward value
        """
        mining_prob = image[self.sight_distance, self.sight_distance]

        reward = mining_prob*self.MINING_REWARD*self.visited[self.__class__.row_position, self.__class__.col_position]
        reward += self.local_target_reached()*self.TARGET_REWARD + self.target_reached()*self.END_REWARD

        if action == 4 or action == 5:
            reward += self.HOVER_PENALTY

        if self.visited[self.__class__.row_position, self.__class__.col_position] == 0:
            reward += self.VISITED_PENALTY
        return reward

    def get_local_map(self):
        """
        Creates local map (shape: 25x25) of mining areas from the region map
        :return: local_map
        """
        local_map = deepcopy(self.__class__.map[self.local_map_lower_row:self.local_map_upper_row+1, self.local_map_lower_col:self.local_map_upper_col+1])
        local_map[(self.local_target[0]-self.local_map_lower_row), (self.local_target[1]-self.local_map_lower_col)] = 1
        return local_map

    def next_local_map(self):
        """
        Sets boundaries on local map, placing the drone at the center of the new map
        :return: void
        """
        self.local_map_lower_row = self.__class__.row_position - 12
        self.local_map_upper_row = self.__class__.row_position + 12
        self.local_map_lower_col = self.__class__.col_position - 12
        self.local_map_upper_col = self.__class__.col_position + 12

        # Corrects for indexing error if drone is too close to a border
        if self.local_map_lower_row < 0:
            self.local_map_lower_row = 0
            self.local_map_upper_row = 24
        if self.local_map_lower_col < 0:
            self.local_map_lower_col = 0
            self.local_map_upper_col = 24
        if self.local_map_upper_row >= 180:
            self.local_map_upper_row = 179
            self.local_map_lower_row = 155
        if self.local_map_upper_col >= 180:
            self.local_map_upper_col = 179
            self.local_map_lower_col = 155

        self.get_local_target()

    def target_reached(self):
        """
        :return: boolean, true if drone is at the target position
        """
        target = False
        if self.current_target[0] - 2 <= self.__class__.row_position <= self.current_target[0] + 2 and \
                self.current_target[1] - 2 <= self.__class__.col_position <= self.current_target[1] - 2:
            target = True
            print(self.current_target, 'reached')

        return target

    def local_target_reached(self):
        """
        :return: boolean, true if drone is at the local target position
        """
        target = False
        if self.__class__.row_position == self.local_target[0] and self.__class__.col_position == self.local_target[1]:
            target = True
        return target

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
        if self.__class__.row_position == target[0]:
            row = self.__class__.row_position - self.local_map_lower_row
        elif self.__class__.row_position > target[0]:
            if self.__class__.row_position - target[0] > 12:
                row = 0
            else:
                row = target[0] - self.__class__.row_position + 12
        elif self.__class__.row_position < target[0]:
            if target[0] - self.__class__.row_position > 12:
                row = 24
            else:
                row = target[0] - self.__class__.row_position+ 12
        if self.__class__.col_position == target[1]:
            col = self.__class__.col_position - self.local_map_lower_col
        elif self.__class__.col_position > target[1]:
            if self.__class__.col_position - target[1] > 12:
                col = 0
            else:
                col = target[1] - self.__class__.col_position + 12
        elif self.__class__.col_position < target[1]:
            if target[1] - self.__class__.col_position > 12:
                col = 24
            else:
                col = target[1] - self.__class__.col_position + 12
        row = int(row+self.local_map_lower_row)
        col = int(col+self.local_map_lower_col)
        self.local_target = [row, col]

    def flatten_state(self, state):
        """
        :param state: 2d array of mining probabilities within the drone's vision
        :return: one dimensional array of the state information
        """
        flat_state = state.reshape(1, self.vision_size)
        return flat_state

    def step_local(self, action, time):
        self.done = False
        next_row = self.__class__.row_position
        next_col = self.__class__.col_position

        # Drone not allowed to move outside of the current local map
        if action == 0:
            if self.__class__.row_position < self.local_map_upper_row and self.__class__.row_position < (
                    self.totalRows - self.sight_distance - 1):  # Forward one grid
                next_row = self.__class__.row_position + 1
                next_col = self.__class__.col_position
            else:
                action = 5
        elif action == 1:
            if self.__class__.col_position < self.local_map_upper_col and self.__class__.col_position < (
                    self.totalCols - self.sight_distance - 1):  # right one grid
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position + 1
            else:
                action = 5
        elif action == 2:
            if self.__class__.row_position > self.local_map_lower_row and self.__class__.row_position > self.sight_distance + 1:  # back one grid
                next_row = self.__class__.row_position - 1
                next_col = self.__class__.col_position
            else:
                action = 5
        elif action == 3:
            if self.__class__.col_position > self.local_map_lower_col and self.__class__.col_position > self.sight_distance + 1:  # left one grid
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position - 1
            else:
                action = 5
        if action == 5 or action == 4:  # This hardcodes the drone to move towards the target if it tries to take an invalid action
            if self.__class__.row_position < self.local_map_upper_row and self.__class__.row_position < (
                    self.totalRows - self.sight_distance - 1) and self.__class__.row_position < self.local_target[0]:
                next_row = self.__class__.row_position + 1
                next_col = self.__class__.col_position
            elif self.__class__.col_position < self.local_map_upper_col and self.__class__.col_position < (
                    self.totalCols - self.sight_distance - 1) and self.__class__.col_position < self.local_target[1]:
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position + 1
            elif self.__class__.row_position > self.local_map_lower_row and self.__class__.row_position > \
                    self.sight_distance + 1 and self.__class__.row_position > self.local_target[0]:
                next_row = self.__class__.row_position - 1
                next_col = self.__class__.col_position
            elif self.__class__.col_position > self.local_map_lower_col and self.__class__.col_position > \
                    self.sight_distance + 1 and self.__class__.col_position > self.local_target[1]:
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position - 1

        self.__class__.row_position = next_row
        self.__class__.col_position = next_col
        # print(self.row_position, self.col_position)
        # print(self.local_target)

        image = self.get_classified_drone_image()

        reward = self.get_reward(image, action)

        self.visited_position()
        self.update_map(image)

        if time > self.config.max_steps_search or self.local_target_reached():
            self.done = True

        flattened_local_map = self.local_map.reshape(1, 1, 625)

        state = self.flatten_state(image)
        state = np.append(state, self.visited[self.__class__.row_position + 1, self.__class__.col_position])
        state = np.append(state, self.visited[self.__class__.row_position, self.__class__.col_position + 1])
        state = np.append(state, self.visited[self.__class__.row_position - 1, self.__class__.col_position])
        state = np.append(state, self.visited[self.__class__.row_position, self.__class__.col_position + 1])
        state = np.append(state, self.local_target[0] - self.__class__.row_position)
        state = np.append(state, self.local_target[1] - self.__class__.col_position)
        state = np.reshape(state, [1, 1, self.vision_size + 6])

        return state, flattened_local_map, reward, self.done
