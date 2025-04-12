import numpy as np
import random as rd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque



BATCH_SIZE = 16



replay_buffer = deque(maxlen=2)
for i in range(20):
    vector = [a + b for a, b in zip([1, 2, 3, 4, 5], [10 * i] * 5)]
    replay_buffer.append(vector)

def sample_experiences():
    indices = np.random.randint(len(replay_buffer), size=BATCH_SIZE)
    batch = [replay_buffer[index] for index in indices]
    return [[experience[field_index] for experience in batch] for field_index in range(5)]

print(replay_buffer)
# print(sample_experiences())





# class Memory:
#     def __init__(self):
#         self.memory = deque(maxlen=10000)
    
#     def push(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
    
#     def sample(self):
#         indices = np.random.randint(len(self.memory), size=BATCH_SIZE)
#         batch = [self.memory[index] for index in indices]
#         return [[experience[field_index] for experience in batch] for field_index in range(5)]
    
#     def __len__(self):
#         return len(self.memory)
    
    

# memory = Memory()
# for i in range(20):
#     vector = [a + b for a, b in zip([1, 2, 3, 4, 5], [10 * i] * 5)]
#     memory.push(vector[0], vector[1], vector[2], vector[3], vector[4])
    

# print(memory.sample())