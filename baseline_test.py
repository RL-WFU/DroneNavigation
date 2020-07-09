from Environment.baseline_env import *

# Preset drone path
actions = []
for r in range(90):
    for i in range(180):
        actions.append(3)
    actions.append(2)
    for i in range(180):
        actions.append(1)
    actions.append(2)

Drone = Baseline()
index = 0
while Drone.calculate_covered('mining') < .7:
    Drone.step(actions[index])
    index += 1
    Drone.plot_path('test_path.jpg')
    Drone.save_map('test_map.jpg')

Drone.plot_path('test_path.jpg')
Drone.save_map('test_map.jpg')
print('Number of Steps Taken:', index)

# Takes 23419 steps to cover 70% of the mining
