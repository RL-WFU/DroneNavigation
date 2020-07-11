from ddrqn import *
from Environment.search_env import *
from Environment.tracing_env import *
from Environment.target_selector_env import *
from Training.training_helper import *

"""
Trains the searching network to navigate to target

For best results: train search and trace networks first, then train with target_cost=True (use cost function to 
select next target), then train full network 

Saves plotting to Training_results/Search and Training_results/Trace and Training_results/Target
"""


def train_full_model(target_cost=False, search_weights=None, trace_weights=None, target_weights=None):
    # Initialize environment and ddrqn agents
    search = Search()
    trace = Trace()
    target = SelectTarget()
    action_size = search.num_actions
    searching_agent = DDRQNAgent(search.vision_size+6, action_size)
    if search_weights is not None:
        searching_agent.load(search_weights + '.h5', search_weights + '_target.h5')

    tracing_agent = DDRQNAgent(trace.vision_size + 4, action_size)
    if trace_weights is not None:
        tracing_agent.load(trace_weights + '.h5', trace_weights + '_target.h5')

    if not target_cost:
        selection_agent = DDRQNAgent(config.num_targets * 3, config.num_targets)
        if target_weights is not None:
            selection_agent.load(target_weights + '.h5', target_weights + '_target.h5')

    done = False
    batch_size = 32

    # Initialize episode logging
    search_rewards = []
    search_covered = []
    search_steps = []
    average_over = int(config.num_episodes / 10)
    search_average_rewards = []
    search_average_r = deque(maxlen=average_over)
    search_episode_num = 0

    trace_rewards = []
    trace_covered = []
    trace_steps = []
    trace_average_rewards = []
    trace_average_r = deque(maxlen=average_over)
    trace_episode_num = 0

    if not target_cost:
        target_selection_rewards = []
        target_selection_average_rewards = []
        target_selection_average_r = deque(maxlen=average_over)
        iteration = 0

    for e in range(config.num_episodes):
        mining_coverage = []
        search.reset_env()
        trace.reset_env()
        target.reset_env()
        t = 0

        if not target_cost:
            episode = []
            Transition = collections.namedtuple("Transition", ["state", "local_map", "action", "reward", "next_state",
                                                               "next_local_map", "done"])
            target_selection_reward = 0
            target_selection_state = np.zeros([1, 27])

        while target.calculate_covered('mining') < .7:
            mining = target.calculate_covered('mining')
            print('Mining Coverage:', mining)
            mining_coverage.append(mining)
            print('Total Steps:', t)

            # Complete one searching episode
            reward, steps, row_position, col_position = search_episode(search, searching_agent, batch_size,
                                                                       trace.row_position, trace.col_position)
            search_rewards.append(reward)
            search_covered.append(search.calculate_covered('mining'))
            search_steps.append(steps)

            search_average_r.append(reward)

            if search_episode_num < average_over:
                r = 0
                for i in range(search_episode_num):
                    r += search_average_r[i]
                r /= (search_episode_num + 1)
                search_average_rewards.append(r)
            else:
                search_average_rewards.append(sum(search_average_r) / average_over)

            if search_episode_num % average_over == 0:
                save_plots(search_episode_num + 1, search, 'Search', search_average_rewards, search_rewards,
                           mining_coverage=search_covered)

            save_plots(search_episode_num + 1, search, 'Search')
            print("search episode: {} - {}, reward: {}, mining covered: {}, start position: {},{}, number of steps: {}"
                  .format((search_episode_num + 1) % (e+1), e+1, reward, search_covered[search_episode_num],
                          trace.row_position, trace.col_position, steps))

            search_episode_num += 1
            t += steps

            # Update all environments
            trace.update_visited(search.visited)
            trace.transfer_map(search.map)
            target.update_visited(search.visited)
            target.transfer_map(search.map)

            # Complete one tracing episode
            reward, steps, row_position, col_position = trace_episode(trace, tracing_agent, batch_size,
                                                                      trace.row_position, trace.col_position, target)
            trace_rewards.append(reward)
            trace_covered.append(trace.calculate_covered('mining'))
            trace_steps.append(steps)

            trace_average_r.append(reward)

            if trace_episode_num < average_over:
                r = 0
                for i in range(trace_episode_num):
                    r += trace_average_r[i]
                r /= (trace_episode_num + 1)
                trace_average_rewards.append(r)
            else:
                trace_average_rewards.append(sum(trace_average_r) / average_over)

            if trace_episode_num % average_over == 0:
                save_plots(trace_episode_num + 1, trace, 'Trace', trace_average_rewards, trace_rewards,
                           mining_coverage=trace_covered)

            save_plots(search_episode_num + 1, trace, 'Trace')
            print("trace episode: {} - {}, reward: {}, mining covered: {}, start position: {},{}, number of steps: {}"
                  .format(trace_episode_num+1 % (e+1), e+1, reward, trace_covered[trace_episode_num],
                          search.row_position, search.col_position, steps))

            trace_episode_num += 1
            t += steps

            # Update all environments
            search.update_visited(trace.visited)
            search.transfer_map(trace.map)
            target.update_visited(trace.visited)
            target.transfer_map(trace.map)

            if target_cost:
                next_target = target.select_next_target(trace.row_position, trace.col_position)

            else:
                states = np.zeros([1, 5, 27])
                local_maps = np.zeros([1, 5, 625])
                if iteration < 5:
                    action = target.select_next_target(trace.row_position, trace.col_position)
                else:
                    states, local_maps = get_last_t_states(5, episode, config.num_targets*3)
                    action = selection_agent.act(states, local_maps)

                next_target, next_state, reward = target.set_target(action)
                target_selection_reward += reward

                episode.append(Transition(
                    state=target_selection_state, local_map=trace.local_map, action=action, reward=reward,
                    next_state=next_state, next_local_map=trace.local_map, done=done))

                target_selection_state = next_state

                if iteration > 5:
                    next_states, next_local_maps = get_last_t_states(5, episode, config.num_targets*3)
                    selection_agent.memorize(states, local_maps, action, reward, next_states, next_local_maps, done)

                if target.calculate_covered('mining') >= .7:
                    selection_agent.update_target_model()

                if len(selection_agent.memory) > batch_size:
                    selection_agent.replay(batch_size)

                iteration += 1

            # Update all environments
            search.update_target(next_target)
            trace.update_target(next_target)
            target.update_target(next_target)
            print("Next target:", next_target)

        if not target_cost:
            target_selection_rewards.append(target_selection_reward)
            target_selection_average_r.append(target_selection_reward)

            if e < average_over:
                r = 0
                for i in range(e):
                    r += target_selection_average_r[i]
                    r /= (trace_episode_num + 1)
                target_selection_average_rewards.append(r)
            else:
                target_selection_average_rewards.append(sum(target_selection_average_r) / average_over)

            if e % average_over == 0:
                save_plots(e + 1, target, 'Target', target_selection_average_rewards, target_selection_reward)

            print("trace episode: {}/{}, reward: {}".format(e+1, config.num_episodes, sum(target_selection_rewards)))

        print("EPISODE {} COMPLETE: Steps -- {}, Mining Coverage -- {}, Total Coverage: {}"
              .format(e+1, t, target.calculate_covered('mining'), target.calculate_covered('map')))

        save_weights(e + 1, search, 'search_full_model_weights')
        save_weights(e + 1, trace, 'trace_full_model_weights')
        if not target_cost:
            save_weights(e + 1, target, 'target_selection_full_model_weights')
