from env import *
from configurationSimple import ConfigSimple as config

env = Env(config)

actions = []
for l in range(200):
    actions.append(0)
    actions.append(4)
    actions.append(1)

total_reward = 0
for a in actions:
    state, local_map, reward, done = env.step(a, 1)
    total_reward += reward
    env.save_local_map('local_map_test.png')

print('Reward:', total_reward)
env.plot_path('drone_path_test.png')
env.save_map('map_test.png')