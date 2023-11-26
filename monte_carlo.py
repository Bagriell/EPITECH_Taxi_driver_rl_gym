import gymnasium as gym
import numpy as np
import random
from os import system, name
from time import sleep
from tqdm import tqdm

# Define function to clear console window.
def clear(): 
  
    # Clear on Windows.
    if name == 'nt': 
        _ = system('cls')
  
    # Clear on Mac and Linux. (os.name is 'posix') 
    else: 
        _ = system('clear')

clear()

"""Training the Agent"""

def epsilon_greedy(env, Qtable, state, epsilon):
    # Generate a random number and compare to epsilon, if lower then explore, otherwise exploit
    randnum = np.random.uniform(0, 1.0)
    if randnum < epsilon:
        action = env.action_space.sample()    # explore
    else:
        action = np.argmax(Qtable[state])  # exploit
    return action

#Mettre mine epsilon a 0 / start epsilon a 0 sinon sa ne marche pas, je ne sais pas pourquoi
def monte_carlo(env, learning_rate: float, epoch: int, start_epsilon: float, decay_rate: float, final_epsilon: float, q_table = np.zeros([]), n_table = np.zeros([])):
    gym.wrappers.RecordEpisodeStatistics(env, deque_size=epoch)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    q_table[env.observation_space.n - 1] = 0
    gamma = 0.6
    for i in tqdm(range(epoch)):
        state, info = env.reset()
        done = False
        epsilon = max(final_epsilon, start_epsilon - (i * decay_rate))
        while not done:
            action = np.argmax(q_table[state, :] + np.random.randn(1, env.action_space.n) * epsilon)
            next_state, reward, done, info, _ = env.step(action)
            q_table[state, action] = reward + gamma * np.max(q_table[next_state])
            state = next_state
    return q_table
