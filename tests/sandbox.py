import math
import time
import os, sys
import numpy as np
import random as rd
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)
# from dqn_trainer import DeepQNetwork
import matplotlib.pyplot as plt

rd.seed(42)




# # vec = np.random.random((1, 5))
# vec = np.random.random((16, 5))
# print(vec.shape)

# dqn = DeepQNetwork()
# print(dqn.model.predict(vec).shape)




x = np.arange(1100)
y = -11.5 + np.random.randn(1100) * (1 / (x + 1)**0.3) + np.log(x + 1) / 5

def window_average(y, win):
    return [sum(y[i : i+win]) / win for i in range(len(y) - win)]

def moving_average(rewards_per_episode):
    len_window = 100
    moyenne_mobile = []
    for i in range(len(rewards_per_episode)):
        if i < len_window:
            start_index = 0
        else:
            start_index = i - len_window + 1
        window = rewards_per_episode[start_index: i + 1]
        moyenne_mobile.append(sum(window) / len(window))
    return moyenne_mobile


win = 100
plt.figure()
plt.plot(y)
plt.plot(x, moving_average(y, win), color='black')
x = np.arange(int(win / 2), len(y)-int(win / 2))
plt.plot(x, window_average(y))
plt.title("Rewards sum per episode")
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.show()






