import math
import time
import os, sys
import numpy as np
import random as rd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from dqn_trainer import DeepQNetwork

rd.seed(42)




# vec = np.random.random((1, 5))
vec = np.random.random((16, 5))
print(vec.shape)

dqn = DeepQNetwork()
print(dqn.model.predict(vec).shape)


