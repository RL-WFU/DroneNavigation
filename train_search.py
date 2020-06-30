from configurationSimple import ConfigSimple as config
from ddqn import *
import matplotlib.pyplot as plt
from env import Env as Drone

# Initialize environment and ddqn agent
env = Drone(config)
action_size = env.num_actions
agent = DDQNAgent(env.vision_size, action_size)

done = False
batch_size = 32

episode_rewards = []
episode_covered = []
episode_steps = []
average_over = config.num_episodes / 10
average_rewards = []
average_covered = []
average_r = 0
average_c = 0

for e in range(config.num_episodes):
    total_reward = 0
    episode_rewards.append(0)
    state, local_map = env.reset_environment()
    t = 0
    for time in range(config.max_steps):
        action = agent.act(state, local_map)
        next_state, next_local_map, reward, done = env.step(action, time)
        total_reward += reward

        agent.memorize(state, local_map, action, reward, next_state, next_local_map, done)
        state = next_state
        local_map = next_local_map

        if done:
            agent.update_target_model()
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        t = time

    episode_steps.append(t)
    episode_rewards[e] = total_reward
    covered = env.calculate_covered('local')
    episode_covered.append(covered)
    average_r += total_reward
    average_c += covered

    if e % average_over == 0:
        average_r /= average_over
        average_c /= average_over
        average_rewards.append(average_r)
        average_covered.append(average_c)
        average_r = 0
        average_c = 0

        plt.plot(episode_rewards)
        plt.ylabel('Episode reward')
        plt.xlabel('Episode')
        plt.savefig('ddqn_reward.png')
        plt.clf()

        plt.ylabel('Percent Covered')
        plt.xlabel('Episode')
        plt.savefig('ddqn_coverage.png')
        plt.clf()

        plt.plot(episode_steps)
        plt.ylabel('Steps Taken')
        plt.xlabel('Episode')
        plt.savefig('ddqn_steps.png')
        plt.clf()

    env.plot_path('ddqn_drone_path.jpg')
    env.save_map('ddqn_map.jpg')
    env.save_local_map('ddqn_local_map.jpg')

    print("episode: {}/{}, reward: {}, percent covered: {}, start position: {},{}, number of steps: {}"
            .format(e, config.num_episodes, total_reward, covered, env.start_row,
                    env.start_col, t))

plt.plot(average_rewards)
plt.ylabel('Averaged Episode reward')
plt.xlabel('Episode')
plt.savefig('ddqn_average_reward.png')
plt.clf()

plt.plot(average_covered)
plt.ylabel('Average Percent Covered')
plt.xlabel('Episode')
plt.savefig('ddqn_average_coverage.png')
plt.clf()

plt.plot(episode_rewards)
plt.ylabel('Episode reward')
plt.xlabel('Episode')
plt.savefig('ddqn_reward.png')
plt.clf()

plt.ylabel('Percent Covered')
plt.xlabel('Episode')
plt.savefig('ddqn_coverage.png')
plt.clf()

