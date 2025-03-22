import numpy as np
import math
import matplotlib.pyplot as plt




NUM_EPISODES = 600

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = int(NUM_EPISODES * 2 / 5)

EPS_MIN = 0.1
EPS_FACTOR = int(NUM_EPISODES * 5 / 6) 


epsilon1 = []
epsilon2 = []

for episode in range(NUM_EPISODES):
    print(episode)
    epsilon1.append(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY))
    epsilon2.append(max(1 - episode / EPS_FACTOR, EPS_MIN))


plt.figure(figsize=(8, 4))
plt.plot(epsilon1)
plt.plot(epsilon2)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.ylim(bottom=0)
plt.grid(True)
plt.show()


