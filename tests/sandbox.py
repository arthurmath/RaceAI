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


# x_space = np.linspace(0, 1200, 30)
# print(np.digitize(1000, x_space))



# L1 = np.arange(50)
L2 = np.arange(50, 100, -1)

# plt.figure(1)
# plt.subplot(121) # plot on a 1 row * 2 col grid, at cell 1
# plt.plot(L1)
# plt.subplot(122) # cell 2
# plt.plot(L2)
# plt.savefig(f'raceAI_dql_.png')




x = np.arange(1100)
y = -11.5 + np.random.randn(1100) * (1 / (x + 1)**0.3) + np.log(x + 1) / 5

def window_average(y, win):
    return [sum(y[i : i+win]) / win for i in range(len(y) - win)]

win = 100
average = window_average(y, win)

# plt.figure()
# plt.plot(x, y)
# plt.plot(x[int(win / 2):-int(win / 2)], average, color='black')
# plt.tight_layout()
# plt.show()



win = 100
x = np.arange(int(win / 2), len(y)-int(win / 2))
plt.figure()
plt.plot(y)
plt.plot(x, window_average(y, win), color='black')
plt.title("Rewards sum per episode")
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.show()