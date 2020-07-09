from Environment.base_env import *


class SelectTarget(Env):
    def __init__(self):
        # Set simulation
        self.set_simulation_map()

        # Define initial targets
        self.__class__.current_target_index = 0
        self.current_target = self.targets[self.__class__.current_target_index]

        # Set task-specific parameters
        self.num_actions = 9
        self.region_values = np.zeros([9, 3])

        # Set reward values
        self.MINING_REWARD = 100
        self.DISTANCE_PENALTY = -2
        self.COVERED_PENALTY = -1200
        self.HOVER_PENALTY = -75

    def set_target(self, next_target):
        self.done = False
        self.current_target = self.targets[next_target]

        self.update_regions()

        reward = self.get_reward(next_target)

        self.__class__.current_target_index = next_target

        if self.calculate_covered('global') > 0.6:
            self.done = True

        state = self.regions.reshape(1, 27)

        return self.current_target, state, reward, self.done

    def get_reward(self, next_target):
        hover = False
        if next_target == self.__class__.current_target_index:
            hover = True

        reward = self.region_values[next_target, 0]*self.MINING_REWARD + self.region_values[next_target, 1]*self.COVERED_PENALTY + \
            self.region_values[next_target, 2]*self.DISTANCE_PENALTY

        return reward

    def update_regions(self):
        self.region_values[:, 0] = self.get_mining()
        self.region_values[:, 1] = self.get_covered()
        self.region_values[:, 2] = self.get_distance()

    def get_mining(self):
        mining = np.zeros(9)

        for i in range(9):
            v = 0
            for j in range(int(self.totalRows/3)):
                for k in range(int(self.totalCols/3)):
                    if self.map[int(j+self.regions[i][0]), int(k+self.regions[i][1])] > 0:
                        mining[i] += 1
                    if self.visited[int(j+self.regions[i][0]), int(k+self.regions[i][1])] < 1:
                        v += 1
            mining[i] /= v+1

        return mining

    def get_covered(self):
        covered = np.zeros(9)
        for i in range(9):
            for j in range(int(self.totalRows/3)):
                for k in range(int(self.totalRows/3)):
                    if self.visited[int(j+self.regions[i][0]), int(k+self.regions[i][1])] < 1:
                        covered[i] += 1
            covered[i] = covered[i]/((self.totalRows/3)**2)

        return covered

    def get_distance(self):
        distance = np.zeros(9)
        for i in range(9):
            distance[i] = math.sqrt((self.__class__.row_position - self.targets[i][0])**2 +
                                    (self.__class__.col_position - self.targets[i][1])**2)

        return distance

    def select_next_target(self, row, col):
        self.__class__.row_position = row
        self.__class__.col_position = col
        self.update_regions()
        next_targets = np.zeros(9)
        for i in range(9):
            next_targets[i] = self.region_values[i, 0]*self.MINING_REWARD + self.region_values[i, 1]*self.COVERED_PENALTY + \
                self.region_values[i, 2]*self.DISTANCE_PENALTY
            if i == self.current_target_index:
                next_targets[i] += self.HOVER_PENALTY

        # print(self.region_values)
        return np.argmax(next_targets)

    def simple_select(self):
        next_target = self.current_target_index + 1
        return next_target
