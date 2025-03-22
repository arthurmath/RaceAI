import numpy as np
import math
import matplotlib.pyplot as plt



N_EPISODES = 2
N_STEPS = 100    
EPISODE_INCREASE = 2

MUTATION_RATE = 0.9
MR_MIN = 0.2
MR_FACTOR = int(N_EPISODES * 5 / 6) 




for generation in range(N_EPISODES):
    mutation_rate = max(1 - generation / MR_FACTOR, MR_MIN)

    print(mutation_rate)


