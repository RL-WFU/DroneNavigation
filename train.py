from configurationSimple import ConfigSimple as config
from ddrqn import *
import matplotlib.pyplot as plt
from Environment.search_env import *
from Environment.target_selector_env import *
from Environment.tracing_env import *
import collections


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


# Initialize environment and ddqn agent
search = Search()
trace = Trace()
target = SelectTarget()
action_size = search.num_actions
searching_agent = DDRQNAgent(search.vision_size+6, action_size)
searching_agent.load('full_searching_model_weights_mid.h5', 'full_searching_target_model_weights_mid.h5')
tracing_agent = DDRQNAgent(trace.vision_size + 4, action_size)
tracing_agent.load('full_tracing_model_weights.h5', 'full_tracing_target_model_weights.h5')
selection_agent = DDRQNAgent(config.num_targets*3, config.num_targets)

done = False
batch_size = 32

Transition = collections.namedtuple("Transition", ["state", "local_map", "action", "reward", "next_state", "next_local_map", "done"])
episode_rewards = []
episode_covered = []
episode_steps = []
average_over = 1
average_rewards = []
average_r = deque(maxlen=average_over)
average_c = 0
region_values = []

for e in range(config.num_episodes):
    episode = []
    mining_coverage = []
    total_reward = 0
    episode_rewards.append(0)
    search.reset_env()
    trace.reset_env()
    target.reset_env()
    steps = 0

    while target.calculate_covered('mining') < .7:
        t = 0
        mining = target.calculate_covered('mining')
        print('Mining Coverage:', mining)
        mining_coverage.append(mining)
        print('Total Steps:', steps)

        state, local_map = search.reset_search(search.row_position, search.col_position)

        for time in range(config.max_steps_search):
            states = np.zeros([1, 5, 29])
            local_maps = np.zeros([1, 5, 625])
            if time < 5:
                action = np.random.randint(0, 5)
            else:
                states, local_maps = get_last_t_states(5, episode, search.vision_size+6)
                action = searching_agent.act(states, local_maps)

            next_state, next_local_map, reward, done = search.step(action, time)
            total_reward += reward

            episode.append(Transition(
                state=state, local_map=local_map, action=action, reward=reward, next_state=next_state, next_local_map=next_local_map, done=done))

            state = next_state
            local_map = next_local_map
            if time > 5:
                next_states, next_local_maps = get_last_t_states(5, episode, search.vision_size+6)
                searching_agent.memorize(states, local_maps, action, reward, next_states, next_local_maps, done)

            if done:
                searching_agent.update_target_model()
                break

            if len(searching_agent.memory) > batch_size:
                searching_agent.replay(batch_size)

            t = time

        steps += t

        search.save_local_map('ddrqn_local_map.jpg')
        search.plot_path('ddrqn_drone_path.jpg')
        search.save_map('ddrqn_map.jpg')
        trace.update_visited(search.visited)
        trace.transfer_map(search.map)
        target.update_visited(search.visited)
        target.transfer_map(search.map)

        state, local_map = trace.reset_tracing(search.row_position, search.col_position)
        coverage = trace.calculate_covered('mining')
        for time in range(config.max_steps_trace):
            states = np.zeros([1, 5, 29])
            local_maps = np.zeros([1, 5, 625])
            if time < 5:
                action = np.random.randint(0, 5)
            else:
                states, local_maps = get_last_t_states(5, episode, trace.vision_size+4)
                action = tracing_agent.act(states, local_maps)

            next_state, next_local_map, reward, done = trace.step(action, time)
            total_reward += reward

            episode.append(Transition(
                state=state, local_map=local_map, action=action, reward=reward, next_state=next_state, next_local_map=next_local_map, done=done))

            state = next_state
            local_map = next_local_map
            if time > 5:
                next_states, next_local_maps = get_last_t_states(5, episode, trace.vision_size+4)
                tracing_agent.memorize(states, local_maps, action, reward, next_states, next_local_maps, done)

            if done:
                tracing_agent.update_target_model()
                break

            if len(tracing_agent.memory) > batch_size:
                tracing_agent.replay(batch_size)

            if (time + 1) % 100 == 0:
                new_coverage = trace.calculate_covered('mining')
                if new_coverage - coverage < .005:
                    next_target = target.select_next_target(trace.row_position, trace.col_position)
                    if next_target != trace.current_target_index:
                        break
                coverage = new_coverage

                tracing_agent.save('full_tracing_model_weights_test.h5', 'full_tracing_target_model_weights_test.h5')

            t = time

        steps += t

        search.update_visited(trace.visited)
        search.transfer_map(trace.map)
        target.update_visited(trace.visited)
        target.transfer_map(trace.map)
        search.save_local_map('ddrqn_local_map.jpg')
        search.plot_path('ddrqn_drone_path.jpg')
        search.save_map('ddrqn_map.jpg')

        # next_target = target.simple_select()
        next_target = target.select_next_target(trace.row_position, trace.col_position)
        region_values.append(target.region_values)
        search.update_target(next_target)
        trace.update_target(next_target)
        target.update_target(next_target)
        print("Next target:", next_target)

    episode_steps.append(steps)
    episode_rewards[e] = total_reward
    covered = search.calculate_covered('global')
    episode_covered.append(covered)
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

    plt.plot(episode_steps)
    plt.ylabel('Steps Taken')
    plt.xlabel('Episode')
    plt.savefig('ddrqn_steps.png')
    plt.clf()

    plt.plot(mining_coverage)
    plt.ylabel('Iteration')
    plt.xlabel('Episode Mining Coverage')
    plt.savefig('mining_coverage.png')
    plt.clf()

    if e == 0:
        tracing_agent.save('full_tracing_model_weights_mid1.h5', 'full_tracing_target_model_weights_mid1.h5')
        searching_agent.save('full_searching_model_weights_mid1.h5', 'full_searching_target_model_weights_mid1.h5')
        search.plot_path('ddrqn_drone_path1.jpg')
        search.save_map('ddrqn_map1.jpg')
        plt.plot(mining_coverage)
        plt.ylabel('Iteration')
        plt.xlabel('Episode Mining Coverage')
        plt.savefig('mining_coverage1.png')
        plt.clf()

    if e == 1:
        tracing_agent.save('full_tracing_model_weights_mid2.h5', 'full_tracing_target_model_weights_mid2.h5')
        searching_agent.save('full_searching_model_weights_mid2.h5', 'full_searching_target_model_weights_mid2.h5')
        search.plot_path('ddrqn_drone_path2.jpg')
        search.save_map('ddrqn_map2.jpg')
        plt.plot(mining_coverage)
        plt.ylabel('Iteration')
        plt.xlabel('Episode Mining Coverage')
        plt.savefig('mining_coverage2.png')
        plt.clf()

    if e == 2:
        tracing_agent.save('full_tracing_model_weights_mid3.h5', 'full_tracing_target_model_weights_mid3.h5')
        searching_agent.save('full_searching_model_weights_mid3.h5', 'full_searching_target_model_weights_mid3.h5')
        search.plot_path('ddrqn_drone_path3.jpg')
        search.save_map('ddrqn_map3.jpg')
        plt.plot(mining_coverage)
        plt.xlabel('Iteration')
        plt.ylabel('Episode Mining Coverage')
        plt.savefig('mining_coverage2.png')
        plt.clf()

    print("episode: {}/{}, reward: {}, percent covered: {}, start position: {},{}, number of steps: {}"
            .format(e, config.num_episodes, total_reward, covered, search.start_row,
                    search.start_col, steps))

print('========== Values for Target Training ==========')
print(region_values)

tracing_agent.save('full_tracing_model_weights.h5', 'full_tracing_target_model_weights.h5')
searching_agent.save('full_searching_model_weights.h5', 'full_searching_target_model_weights.h5')

plt.plot(average_rewards)
plt.ylabel('Averaged Episode reward')
plt.xlabel('Episode')
plt.savefig('ddrqn_average_reward.png')
plt.clf()

plt.plot(episode_rewards)
plt.ylabel('Episode reward')
plt.xlabel('Episode')
plt.savefig('ddrqn_reward.png')
plt.clf()

plt.plot(episode_covered)
plt.ylabel('Percent Covered')
plt.xlabel('Episode')
plt.savefig('ddrqn_coverage.png')
plt.clf()


