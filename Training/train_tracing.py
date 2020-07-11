from ddrqn import *
from Environment.tracing_env import *
from Training.training_helper import *

"""
Trains the tracing network to navigate to follow mining

Saves plotting to Training_results/Trace
"""


def train_tracing_agent(weights=None):
    # Initialize environment and ddrqn agent
    trace = Trace()
    action_size = trace.num_actions

    tracing_agent = DDRQNAgent(trace.vision_size + 6, action_size)
    if weights is not None:
        tracing_agent.load(weights + '.h5', weights + '_target.h5')

    done = False
    batch_size = 32

    # Initialize episode logging
    episode_rewards = []
    episode_covered = []
    episode_steps = []
    average_over = int(config.num_episodes / 10)
    average_rewards = []
    average_r = deque(maxlen=average_over)

    for episode in range(config.num_episodes):
        trace.reset_env()
        reward, steps, row_position, col_position = trace_episode(trace, tracing_agent, batch_size,
                                                                   trace.row_position, trace.col_position)
        episode_rewards.append(reward)
        episode_covered.append(trace.calculate_covered('mining'))
        episode_steps.append(steps)

        average_r.append(reward)

        if episode < average_over:
            r = 0
            for i in range(episode):
                r += average_r[i]
            r /= (episode + 1)
            average_rewards.append(r)
        else:
            average_rewards.append(sum(average_r) / average_over)

        if episode % average_over == 0:
            save_plots(episode+1, trace, 'Trace', average_rewards, episode_rewards, mining_coverage=episode_covered)

        print("episode: {}/{}, reward: {}, percent covered: {}, start position: {},{}, number of steps: {}"
              .format(episode+1, config.num_episodes, reward, episode_covered[episode], trace.start_row,
                      trace.start_col, steps))

    save_weights(1, trace, 'trace_model_weights')

