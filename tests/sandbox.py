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




L1 = np.arange(50)
L2 = np.arange(50, 100)

plt.figure(1)
plt.subplot(121) # plot on a 1 row * 2 col grid, at cell 1
plt.plot(L1)
plt.subplot(122) # cell 2
plt.plot(L2)
plt.savefig(f'raceAI_dql_.png')