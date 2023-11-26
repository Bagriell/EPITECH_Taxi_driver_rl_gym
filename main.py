import sys
import json
import numpy as np
import q
import sarsa
import monte_carlo
import gymnasium as gym
import time
from collections import defaultdict
from multiprocessing import Process

def test(env, matrix):
    observation, info = env.reset()
    done = False
    total_action = 0
    while not done:
        action = np.argmax(matrix[observation])
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation = next_observation
        total_action += 1
    return total_action

def train(env, config):
    with open(config) as json_file:
        config = json.load(json_file)

    if config["algo"] == "q":
        matrix = q.q(env, config["learning_rate"], config["epoch"], config["start_epsilon"], 0.1, config["final_epsilon"])
    elif config["algo"] == "sarsa":
        matrix = sarsa.sarsa(env, config["learning_rate"], config["epoch"], config["start_epsilon"], 0.1, config["final_epsilon"])
    elif config["algo"] == "monte_carlo":
        matrix = monte_carlo.monte_carlo(env, config["learning_rate"], config["epoch"], config["start_epsilon"], 0.1, config["final_epsilon"])    
    np.save(config["path_file"], matrix)

def benchmark():
    
    
    list_algo = ["q", "sarsa", "monte_carlo"]
    list_learning_rate = [0.1, 0.01, 0.001]
    list_epochs = [1000, 2500, 5000, 10000, 25000, 50000]
    list_start_epsilon = [3.0, 1.0, 0.5]
    list_final_epsilon = [0.1, 0.01, 0.001, 0]

    result = []
    for index, algo in enumerate(list_algo):
        result.append([])
        for learning_rate in list_learning_rate:
            #result[algo][f"learning_rate_{learning_rate}"] = {}
            for epochs in list_epochs:
                #result[algo][f"learning_rate_{learning_rate}"][f"epochs_{epochs}"] = {}
                for start_epsilon in list_start_epsilon:
                    #result[algo][f"learning_rate_{learning_rate}"][f"epochs_{epochs}"][f"start_epsilon_{start_epsilon}"] = {}
                    for final_epsilon in list_final_epsilon:
                        # result[algo][f"learning_rate_{learning_rate}"][f"epochs_{epochs}"][f"start_epsilon_{start_epsilon}"][f"final_epsilon_{final_epsilon}"] = {}
                        env = gym.make("Taxi-v3")
                        start_time = time.time()
                        if algo == "q":
                            matrix = q.q(env, learning_rate, epochs, start_epsilon, 0.1, final_epsilon)
                        elif algo == "sarsa":
                            matrix = sarsa.sarsa(env, learning_rate, epochs, start_epsilon, 0.1, final_epsilon)
                        elif algo == "monte_carlo":
                            matrix = monte_carlo.monte_carlo(env, learning_rate, epochs, start_epsilon, 0.1, final_epsilon)
                        end_time = time.time() - start_time
                        env.close()
                        list_total_action = []
                        for i in range(1000):
                            list_total_action.append(test(env, matrix))
                        result[index].append({
                            "algo": algo,
                            "learning_rate": learning_rate,
                            "epochs": epochs,
                            "start_epsilon": start_epsilon,
                            "final_epsilon": final_epsilon,
                            "average_step": sum(list_total_action) / len(list_total_action),
                            "min_step": min(list_total_action),
                            "max_step": max(list_total_action),
                            "time": end_time
                        })
    with open('benchmark2.json', 'w') as f:
        json.dump(result, f)

def main():
    if sys.argv[1] == "benchmark":
        benchmark()
    elif sys.argv[1] == "train":
        env = gym.make("Taxi-v3")
        train(env, sys.argv[2])
        env.close()
    elif sys.argv[1] == "test":
        env = gym.make("Taxi-v3", render_mode='human')
        test(env, np.load(sys.argv[2]))
        env.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 main.py type optional_file_name')
        print('type: benchmark, train, test')
        print("optional_file_name: file name for train or test.")
        print("if test, you have to give a matrix.npy file")
        print("if train, you have to give a config.json file")
        print("A config.json file must have the following format:")
        print("{")
        print("    \"path_file\": \"myfile.npy\",")
        print("    \"algo\": \"q\" or \"sarsa\" or \"monte_carlo\"")
        print("    \"learning_rate\": float,")
        print("    \"epochs\": int,")
        print("    \"start_epsilon\": float,")
        print("    \"final_epsilon\": float")
        print("}")
        exit(1)
    main()
    exit(0)