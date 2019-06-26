import gym
import numpy as np

# create gym environment
env = gym.make('MountainCar-v0')
env.reset()

# use 20 groups/buckets to convert continuous values to discrete values
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# create q table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 1000

# exploration settings
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# function to convert environment state from continuous to discrete so it doesn't take forever
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


# run the game
for episode in range(EPISODES):

    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            # get action from q-table
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()

        # if the simulation did not end yet after the last step - update Q table
        if not done:

            # maximum possible Q value in the next step (for the new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # update q table with the new q value
            q_table[discrete_state + (action,)] = new_q

        # if the simulation has ended, update q_value with reward
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = reward

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
        

env.close()