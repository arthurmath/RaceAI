import math
import numpy as np
import time
import random as rd



population = [1, 2, 3]
ratios = [0.90, 0.1, 0.1]


# for _ in range(10):
#     print(rd.choices(population, weights=ratios, k=2))



# actions = [[False, False, True, False], [True, False, True, False]]

# actions = [[i if act else -1 for i, act in enumerate(action)] for action in actions]
# actions = [[x for x in action if x != -1] for action in actions]

# print(actions)


# actions = [[False, False, True, False], [True, False, True, False]]


# # acts = []
# # for i, action in enumerate(actions):
# #     acts.append([])
# #     for j, act in enumerate(action):
# #         if act:
# #             acts[i].append(j)

# actions = [[j for j, act in enumerate(action) if act] for action in actions]

# print(actions)


nb_cars = 4

actions = [[np.random.choice(4, p=[3/6, 1/6, 1/6, 1/6])] for _ in range(nb_cars)]
print(actions)
actions = [[1 if i == action[0] else 0 for i in range(4)] for action in actions]

print(actions)