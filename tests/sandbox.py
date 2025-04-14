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

'''
x_space = np.linspace(0, 1200, 30)

state_x = np.digitize(1000, x_space)


L1 = np.arange(50)
L2 = np.arange(50, 100)

plt.figure(1)
plt.subplot(121) # plot on a 1 row * 2 col grid, at cell 1
plt.plot(L1)
plt.subplot(122) # cell 2
plt.plot(L2)
plt.savefig(f'raceAI_dql_.png')

'''
#fonction moyenne mobile
rewards_per_episode = [10,20,30,40,50,60,70,80,90]
def moving_average(rewards_per_episode, fenetre):
    moyenne_mobile = []
    for i in range(len(rewards_per_episode)):
        if i < fenetre:
            start_index = 0
        else:
            start_index = i - fenetre + 1
        window = rewards_per_episode[start_index : i+1 ]#on selectionne la fenêtre pour faire la moyenne
        print(window)
        moyenne = sum(window) / len(window)
        print(moyenne)
        moyenne_mobile.append(moyenne)
    return moyenne_mobile
result = moving_average(rewards_per_episode,4)
print(result)