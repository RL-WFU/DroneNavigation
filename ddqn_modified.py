import random
import numpy as np
from collections import deque
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, LSTM
from keras.optimizers import Adam


class DDQNAgent:
    def __init__(self, state_size, action_size, training=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        if not training:
            self.epsilon = self.epsilon_min
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.learning_rate_decay = 0.8
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        first_input = Input(shape=(5, self.state_size))
        layer1 = Dense(64, activation='relu')(first_input)
        layer2 = Dense(10, activation='relu')(layer1)
        layer3 = Dense(self.action_size, activation='linear')(layer2)

        model = Model(inputs=first_input, outputs=layer3)
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
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, local_map, action, reward, next_state, next_local_map, done in minibatch:
            target = self.model.predict(state)
            target_next = self.model.predict(next_state)
            target_val = self.target_model.predict(next_state)
            if done:
                target[0][0][action] = reward
            else:
                a = np.argmax(target_next[0][0])
                target[0][0][action] = reward + self.gamma * target_val[0][0][a]

            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decay_learning_rate(self):
        self.learning_rate *= self.learning_rate_decay

    def load(self, name, name2):
        self.model.load_weights(name)
        self.target_model.load_weights(name2)

    def save(self, name, name2):
        self.model.save_weights(name)
        self.target_model.save_weights(name2)
