import sys
import os
import numpy as np
import random as rd
from pathlib import Path
import pickle

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gene_pilot import Pilot




pilot = Pilot()
print(pilot)


PATH = Path("results_gene/weights")
with open(PATH / Path(f"test.weights"), "wb") as f: # write binary
    pickle.dump((pilot.weights, pilot.bias), f)
    
with open(PATH / Path(f"test.weights"), "rb") as f:
    weights, bias = pickle.load(f)

pilot_saved = Pilot(weights, bias)




states = [[rd.uniform(-1, 1) for _ in range(5)] for _ in range(50)]

for state in states:

    action = pilot.predict(state)
    action_saved = pilot_saved.predict(state)
    
    print(action_saved == action)


