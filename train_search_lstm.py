from configurationSimple import ConfigSimple as config
from ddrqn import *
import matplotlib.pyplot as plt
from env import Env as Drone
import collections

def get_last_t_states(t, episode):
    states = []
    maps = []
    for i, transition in enumerate(episode[-t:]):
        states.append(transition.state)
        maps.append(transition.local_map)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, ((env.sight_distance * 2 + 1) * (env.sight_distance * 2 + 1)) + 6])

    maps = np.asarray(maps)
    maps = np.reshape(maps, [1, t, 25*25])

    return states, maps


def get_last_t_minus_one_states(t, episode):
    states = []
    maps = []
    for i, transition in enumerate(episode[-t + 1:]):
        states.append(transition.state)
        maps.append(transition.local_map)

    states.append(episode[-1].next_state)
    maps.append(episode[-1].next_local_map)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, ((env.sight_distance * 2 + 1) * (env.sight_distance * 2 + 1)) + 6])

    maps = np.asarray(maps)
    maps = np.reshape(maps, [1, t, 25 * 25])

    return states, maps



# Initialize environment and ddqn agent
env = Drone(config)
action_size = env.num_actions
agent = DDRQNAgent(env.vision_size, action_size)

done = False
batch_size = 32

Transition = collections.namedtuple("Transition", ["state", "local_map", "action", "reward", "next_state", "next_local_map", "done"])
episode_rewards = []
episode_covered = []
episode_steps = []
average_over = int(config.num_episodes / 20)
average_rewards = []
average_covered = []
average_r = deque(maxlen=average_over)
average_c = 0

for e in range(config.num_episodes):
    episode = []
    total_reward = 0
    episode_rewards.append(0)
    state, local_map = env.reset_environment()
    t = 0
    for time in range(config.max_steps):
        states = np.zeros([1, 5, 29])
        local_maps = np.zeros([1, 5, 625])
        if t < 5:
            action = np.random.randint(0, 5)
        else:
            states, local_maps = get_last_t_states(5, episode)
            action = agent.act(states, local_maps)

        next_state, next_local_map, reward, done = env.step(action, time)
        total_reward += reward

        episode.append(Transition(
            state=state, local_map=local_map, action=action, reward=reward, next_state=next_state, next_local_map=next_local_map, done=done))

        #agent.memorize(state, local_map, action, reward, next_state, next_local_map, done)
        state = next_state
        local_map = next_local_map
        if time > 5:
            next_states, next_local_maps = get_last_t_states(5, episode)
            agent.memorize(states, local_maps, action, reward, next_states, next_local_maps, done)

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
    average_c += covered
    average_r.append(total_reward)

    if e < average_over:
        r = 0
        for i in range(e):
            r += average_r[i]
        r /= (e+1)
        average_rewards.append(r)
    else:
        average_rewards.append(sum(average_r)/average_over)

    plt.plot(average_rewards)
    plt.ylabel('Averaged Episode reward')
    plt.xlabel('Episode')
    plt.savefig('ddrqn_average_reward.png')
    plt.clf()

    if e % average_over == 0:
        average_c /= average_over
        average_covered.append(average_c)
        average_c = 0

        plt.plot(episode_rewards)
        plt.ylabel('Episode reward')
        plt.xlabel('Episode')
        plt.savefig('ddrqn_reward.png')
        plt.clf()

        plt.ylabel('Percent Covered')
        plt.xlabel('Episode')
        plt.savefig('ddrqn_coverage.png')
        plt.clf()

        plt.plot(episode_steps)
        plt.ylabel('Steps Taken')
        plt.xlabel('Episode')
        plt.savefig('ddrqn_steps.png')
        plt.clf()

    env.plot_path('ddrqn_drone_path.jpg')
    env.save_map('ddrqn_map.jpg')
    env.save_local_map('ddrqn_local_map.jpg')

    print("episode: {}/{}, reward: {}, percent covered: {}, start position: {},{}, number of steps: {}"
            .format(e, config.num_episodes, total_reward, covered, env.start_row,
                    env.start_col, t))

plt.plot(average_rewards)
plt.ylabel('Averaged Episode reward')
plt.xlabel('Episode')
plt.savefig('ddrqn_average_reward.png')
plt.clf()

plt.plot(average_covered)
plt.ylabel('Average Percent Covered')
plt.xlabel('Episode')
plt.savefig('ddrqn_average_coverage.png')
plt.clf()

plt.plot(episode_rewards)
plt.ylabel('Episode reward')
plt.xlabel('Episode')
plt.savefig('ddrqn_reward.png')
plt.clf()

plt.ylabel('Percent Covered')
plt.xlabel('Episode')
plt.savefig('ddrqn_coverage.png')
plt.clf()

