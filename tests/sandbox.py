import math
import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import random as rd




# def my_agent(x):
#     return x*x

# import inspect
# import os

# def write_agent_to_file(function, file):
#     """ copie le code de fonction dans le file """
#     with open(file, "a" if os.path.exists(file) else "w") as f:
#         f.write(inspect.getsource(function))
#         print(function, "written to", file)

# write_agent_to_file(my_agent, "submission.py")



population = [1, 2, 3]
ratios = [0.90, 0.1, 0.1]

population = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002770454230382389, 0.046711341388100014, 0.47155452878089]
ratios = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005342767487661691, 0.09008191990197952, 0.9093838033492543]

for _ in range(10):
    print(rd.choices(population, weights=ratios, k=2))












