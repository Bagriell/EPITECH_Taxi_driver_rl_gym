from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

class TaxiAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
        """
        self.env = env
        self.q_values = np.zeros([env.observation_space.n, env.action_space.n])

        self.lr = learning_rate

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.discount_factor = discount_factor
        self.training_error = []

    def get_action(self, obs: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] += self.lr * temporal_difference

        # update epsilon
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

        # update training error
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self, epsilon_decay: float):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)
    def return_matrix_q(self):
        return self.q_values
    # def set_matrix_q(self, matrix_q):
    #     self.q_values = matrix_q

def q(env, learning_rate: float, epoch: int, start_epsilon: float, epsilon_decay: float, final_epsilon: float, matrix_q = None):
    observation, info = env.reset()
    epsilon_decay = start_epsilon / (epoch / 2)

    agent = TaxiAgent(
        env= env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=epoch)
    for _ in tqdm(range(epoch)):
        observation, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.update(observation, action, reward, terminated, next_observation)
            done = terminated or truncated
            observation = next_observation
        agent.decay_epsilon(epsilon_decay)
    return agent.return_matrix_q()