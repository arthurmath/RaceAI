import numpy as np
import copy as cp
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gene_pilot import Pilot



pilot = Pilot()

pilot1 = pilot

pilot2 = cp.copy(pilot)

pilot3 = cp.deepcopy(pilot)

pilot4 = cp.copy(pilot3)


print(pilot)
print(pilot1)
print(pilot2)
print(pilot3)


for i, weight in enumerate(pilot.weights):
    print((pilot4.weights[i] == weight).all())


