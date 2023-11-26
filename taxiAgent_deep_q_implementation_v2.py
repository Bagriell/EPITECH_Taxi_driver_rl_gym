# An implementation of the Deep Q-Learning algorithm for the Taxi-v2 environment
#
# Path: taxiAgent_deep_q_implementation_v2.py

import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

# Create an deep Q-learning agent
class DeepQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array(state))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        print(minibatch)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            print(next_state)
            self.model.predict(np.array(next_state)[0])
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(np.array(next_state))))
            target_f = self.model.predict(np.array(state))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


env = gym.make("Taxi-v3")
state_size = env.observation_space.n
action_size = env.action_space.n
# print(state_size, action_size)
agent = DeepQAgent(env.observation_space.n, env.action_space.n)

learning_rate = 0.1
epochs = 1000
batch_size = 32
done = False
for e in range(epochs):
    state, info = env.reset()
    #state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.get_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        reward = reward if not done else -10
        #next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, epochs, time, agent.epsilon))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

