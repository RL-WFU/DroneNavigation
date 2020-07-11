from configurationSimple import ConfigSimple as config
import collections
import numpy as np
import matplotlib.pyplot as plt


def search_episode(search, searching_agent, batch_size, row_position, col_position):
    Transition = collections.namedtuple("Transition",
                                        ["state", "local_map", "action", "reward", "next_state", "next_local_map",
                                         "done"])
    episode = []
    total_reward = 0
    t = 0
    state, local_map = search.reset_search(row_position, col_position)

    for time in range(config.max_steps_search):
        states = np.zeros([1, 5, 29])
        local_maps = np.zeros([1, 5, 625])
        if time < 5:
            action = np.random.randint(0, 5)
        else:
            states, local_maps = get_last_t_states(5, episode, search.vision_size + 6)
            action = searching_agent.act(states, local_maps)

        next_state, next_local_map, reward, done = search.step(action, time)
        total_reward += reward

        episode.append(Transition(
            state=state, local_map=local_map, action=action, reward=reward, next_state=next_state,
            next_local_map=next_local_map, done=done))

        state = next_state
        local_map = next_local_map
        if time > 5:
            next_states, next_local_maps = get_last_t_states(5, episode, search.vision_size + 6)
            searching_agent.memorize(states, local_maps, action, reward, next_states, next_local_maps, done)

        if done:
            searching_agent.update_target_model()
            break

        if len(searching_agent.memory) > batch_size:
            searching_agent.replay(batch_size)
        t = time

    return total_reward, t, search.row_position, search.col_position


def trace_episode(trace, tracing_agent, batch_size, row_position, col_position, target=None):
    Transition = collections.namedtuple("Transition",
                                        ["state", "local_map", "action", "reward", "next_state", "next_local_map",
                                         "done"])
    episode = []
    total_reward = 0
    t = 0
    state, local_map = trace.reset_tracing(row_position, col_position)
    coverage = trace.calculate_covered('mining')
    for time in range(config.max_steps_trace):
        states = np.zeros([1, 5, 29])
        local_maps = np.zeros([1, 5, 625])
        if time < 5:
            action = np.random.randint(0, 5)
        else:
            states, local_maps = get_last_t_states(5, episode, trace.vision_size + 4)
            action = tracing_agent.act(states, local_maps)

        next_state, next_local_map, reward, done = trace.step(action, time)
        total_reward += reward

        episode.append(Transition(
            state=state, local_map=local_map, action=action, reward=reward, next_state=next_state,
            next_local_map=next_local_map, done=done))

        state = next_state
        local_map = next_local_map
        if time > 5:
            next_states, next_local_maps = get_last_t_states(5, episode, trace.vision_size + 4)
            tracing_agent.memorize(states, local_maps, action, reward, next_states, next_local_maps, done)

        if done:
            tracing_agent.update_target_model()
            break

        if len(tracing_agent.memory) > batch_size:
            tracing_agent.replay(batch_size)

        if target is not None:
            if (time + 1) % 100 == 0:
                new_coverage = trace.calculate_covered('mining')
                if new_coverage - coverage < .005:
                    next_target = target.select_next_target(trace.row_position, trace.col_position)
                    if next_target != trace.current_target_index:
                        break
                coverage = new_coverage

        t = time

    return total_reward, t, trace.row_position, trace.col_position


def get_last_t_states(t, episode, size):
    states = []
    maps = []
    for i, transition in enumerate(episode[-t:]):
        states.append(transition.state)
        maps.append(transition.local_map)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, size])

    maps = np.asarray(maps)
    maps = np.reshape(maps, [1, t, 25*25])

    return states, maps


def get_last_t_minus_one_states(t, episode, size):
    states = []
    maps = []
    for i, transition in enumerate(episode[-t + 1:]):
        states.append(transition.state)
        maps.append(transition.local_map)

    states.append(episode[-1].next_state)
    maps.append(episode[-1].next_local_map)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, size])

    maps = np.asarray(maps)
    maps = np.reshape(maps, [1, t, 25 * 25])

    return states, maps


def save_weights(num, agent, name):
    agent.save('Training_results/Weights/' + str(name) + '_' + str(num) + '.hf')


def save_plots(num, agent, folder, average_rewards=None, episode_rewards=None, episode_covered=None, mining_coverage=None):
    agent.save_local_map('Training_results/ddrqn_local_map' + str(num) + '.jpg')
    agent.plot_path('Training_results/ddrqn_drone_path' + str(num) + '.jpg')
    agent.save_map('Training_results/ddrqn_map' + str(num) + '.jpg')

    if average_rewards is not None:
        plt.plot(average_rewards)
        plt.ylabel('Averaged Episode reward')
        plt.xlabel('Episode')
        plt.savefig('Training_results/' + str(folder) + '/ddrqn_average_reward.png')
        plt.clf()

    if episode_rewards is not None:
        plt.plot(episode_rewards)
        plt.ylabel('Episode reward')
        plt.xlabel('Episode')
        plt.savefig('Training_results/' + str(folder) + '/ddrqn_reward.png')
        plt.clf()

    if episode_covered is not None:
        plt.plot(episode_covered)
        plt.ylabel('Percent Covered')
        plt.xlabel('Episode')
        plt.savefig('Training_results/ddrqn_coverage.png')
        plt.clf()

    if mining_coverage is not None:
        plt.plot(mining_coverage)
        plt.ylabel('Iteration')
        plt.xlabel('Episode Mining Coverage')
        plt.savefig('Training_results/mining_coverage2.png')
        plt.clf()
