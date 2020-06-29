import random
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input
from keras.optimizers import Adam


class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        #model1 = Sequential()
        #model1.add(Dense(64, input_dim=self.state_size+4, activation='relu'))
        #model1.add(Dense(10, activation='relu'))

        first_input = Input(shape=(1, self.state_size+4))
        layer1 = Dense(64, activation='relu')(first_input)
        layer2 = Dense(10, activation='relu')(layer1)

        #model2 = Sequential()
        #model2.add(Dense(100, input_dim=625, activation= 'relu'))
        #model2.add(Dense(100, activation='relu'))

        second_input = Input(shape=(1, 625))
        layer1_b = Dense(100, activation= 'relu')(second_input)
        layer2_b = Dense(100, activation='relu')(layer1_b)

        merge = concatenate([layer2, layer2_b])
        layer3 = Dense(self.action_size, activation='linear')(merge)
        #model = Sequential()
        #model.add(Concatenate([model1, model2]))
        #model.add(Dense(self.action_size, activation='linear'))

        model = Model(inputs=[first_input, second_input], outputs=layer3)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, local_map, action, reward, next_state, next_local_map, done):
        self.memory.append((state, local_map, action, reward, next_state, next_local_map, done))

    def act(self, state, local_map):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict([state, local_map])
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, local_map, action, reward, next_state, next_local_map, done in minibatch:
            target = self.model.predict([state, local_map])
            target_next = self.model.predict([next_state, next_local_map])
            target_val = self.target_model.predict([next_state, next_local_map])
            if done:
                target[0][0][action] = reward
            else:
                a = np.argmax(target_next[0][0])
                target[0][0][action] = reward + self.gamma * target_val[0][0][a]

            self.model.fit([state, local_map], target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)