from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym
# env = gym.make("LunarLander-v2", render_mode="human")
import csv
import utils
import json

list_learning_rate = [0.1, 0.2, 0.01, 0.001, 0.0001]
list_epoch = [100, 1000, 1500, 2000, 2500, 3000, 5000, 10000, 15000]
benchmark = {}


class TaxiAgent:
    def __init__(
        self,
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
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

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
            return env.action_space.sample()

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
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)

for learning_rate in list_learning_rate:
    benchmark[learning_rate] = {}
    for epoch in list_epoch:
        env = gym.make('Taxi-v3')
        observation, info = env.reset()

        # hyperparameters
        start_epsilon = 1.0
        epsilon_decay = start_epsilon / (epoch / 2)  # reduce the exploration over time
        final_epsilon = 0.1

        env = gym.make('Taxi-v3')
        observation, info = env.reset()

        # hyperparameters
        # learning_rate = 0.1
        # epoch = 5000
        # start_epsilon = 1.0
        # epsilon_decay = start_epsilon / (epoch / 2)  # reduce the exploration over time
        # final_epsilon = 0.1

        # for _ in range(1000):
            
        #     action = env.action_space.sample()  # agent policy that uses the observation and info
        #     observation, reward, terminated, truncated, info = env.step(action)

        #     if terminated or truncated:
        #         observation, info = env.reset()

        agent = TaxiAgent(
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

            agent.decay_epsilon()

            # if _ % 100 == 0:
            #     print(f"Episode {_}: {env.episode_statistics['episode_reward'][-1]}")
        # rolling_length = 500
        # fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        # axs[0].set_title("Episode rewards")
        # # compute and assign a rolling average of the data to provide a smoother graph
        # reward_moving_average = (
        #     np.convolve(
        #         np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        #     )
        #     / rolling_length
        # )
        # axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        # axs[1].set_title("Episode lengths")
        # length_moving_average = (
        #     np.convolve(
        #         np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        #     )
        #     / rolling_length
        # )
        # axs[1].plot(range(len(length_moving_average)), length_moving_average)
        # axs[2].set_title("Training Error")
        # training_error_moving_average = (
        #     np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
        #     / rolling_length
        # )
        # axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        # plt.tight_layout()
        # plt.show()

        env = gym.make('Taxi-v3')
        to_csv = []

        for _ in range(20):
            total_action = 0
            observation, info = env.reset()
            done = False
            while not done:
                action = agent.get_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                observation = next_observation
                total_action += 1
            to_csv.append(total_action)
        print(to_csv)
        moyenne = sum(to_csv) / len(to_csv)
        print(moyenne)
        benchmark[learning_rate][epoch] = moyenne
        # observation, info = env.reset()
        env.close()

# write a json file with the results
with open("benchmark_q_v1.json", "w") as f:
    json.dump(benchmark, f, indent=4)


