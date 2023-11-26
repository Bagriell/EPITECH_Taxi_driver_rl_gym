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

"""Setup"""

 # Setup the Gym Environment

# Make a new matrix filled with zeros.
# The matrix will be 500x6 as there are 500 states and 6 actions.
#Show the user the environment.

"""Training the Agent"""

def epsilon_greedy(env, Qtable, state, epsilon):
    randnum = np.random.uniform(0, 1.0)
    if randnum < epsilon:
        action = env.action_space.sample()    # explore
    else:
        action = np.argmax(Qtable[state])  # exploit
    return action

def sarsa(env, learning_rate: float, epoch: int, start_epsilon:float, decay_rate: float, final_epsilon: float, q_table = np.zeros([])):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    q_table[env.observation_space.n - 1] = 0 


    gamma = 0.5 

    # For plotting metrics
    for i in tqdm(range(epoch)):
        
        state, info = env.reset()
        t = 0
        done = False
        
        epsilon = max(final_epsilon, (start_epsilon - final_epsilon)*np.exp(-decay_rate*i))
        epsilon = 0
        action = epsilon_greedy(env, q_table, state, round(epsilon, 4))
        
        while not done:
            next_state, reward, done, _, info = env.step(action)
            next_action = epsilon_greedy(env, q_table, next_state, epsilon)
            next_q = (reward+gamma*(q_table[next_state][next_action])-q_table[state][action])
            q_table[state][action] = q_table[state][action]+learning_rate*next_q
            state = next_state
            action = next_action
        # print(f"Episode {i+1} finished after {t+1} timesteps. Epsilon value: {epsilon}")
    return q_table