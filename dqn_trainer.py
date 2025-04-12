import os
import time
import pickle
import math
import numpy as np
import random as rd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque



LR = 1e-2
N_STEPS = 200
POPULATION = 1 
BATCH_SIZE = 16 # 32
N_EPISODES = 100 
EPS_FACTOR = int(N_EPISODES * 5 / 6) 
DISCOUNT_FACTOR = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = int(N_EPISODES * 2 / 5)
UPDATE_TARGET = 25
WEIGHTS_PATH = Path("results_dqn/weights")
IMAGES_PATH = Path("results_dqn/images")

SEED = 42
rd.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# print("Devices: ", tf.config.list_physical_devices())




class Memory:
    def __init__(self):
        self.buffer = deque(maxlen=10000)
    
    def push(self, states, actions, next_states, rewards, dones):
        for i in range(len(states)):
            self.buffer.append((states[i], actions[i], next_states[i], rewards[i], dones[i]))
    
    def sample(self):
        indices = np.random.randint(len(self.buffer), size=BATCH_SIZE)
        batch = [self.buffer[index] for index in indices]
        return [[experience[field_index] for experience in batch] for field_index in range(5)]
    
    def __len__(self):
        return len(self.buffer)





class DeepQNetwork:
    def __init__(self):
        self.input_shape = [5]  # = env.observation_space.shape
        self.output_shape = 4   # = env.action_space.n

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(self.input_shape),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(self.output_shape)
        ])
        # print("Nombre de paramètres :", self.model.count_params())
        
        
        
        
        
        


class Trainer:  
    def __init__(self, warm_start):
        self.dqn = DeepQNetwork()
        self.target = tf.keras.models.clone_model(self.dqn.model)  # CHANGED
        self.target.set_weights(self.dqn.model.get_weights())      # CHANGED
        
        if warm_start:
            with open(WEIGHTS_PATH / Path(f"colab1.weights"), "rb") as f:
                self.dqn.model.set_weights(pickle.load(f))
                
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=LR)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        
    def train(self):
        from dqn_game import Session
        self.env = Session(display=True, nb_cars=POPULATION)
        self.memory = Memory()
        self.rewards = []
        self.progressions = []
        self.best_score = 0

        for self.episode in range(N_EPISODES):
            
            self.evaluate_generation()

            if len(self.memory) > BATCH_SIZE:
                self.training_step()
                if self.episode % UPDATE_TARGET == 0:                      # CHANGED
                    self.target.set_weights(self.dqn.model.get_weights())  # CHANGED
                    
            if self.env.quit:
                break
                
            print(f"Ep: {self.episode+1:>2}, progres: {self.progressions[-1]:>4.2f}%, reward: {self.rewards[-1]:>6.2f}, done: {self.step:>3}, epsi:{self.epsilon:.2f}")
            # print()
            
    
    
    
    def evaluate_generation(self):
        
        states = self.env.reset(self.episode)
            
        for self.step in range(N_STEPS + 10 * self.episode):
            actions = self.epsilon_greedy_policy(states)
            moves = [[1 if actions[0] == i else 0 for i in range(4)]]
            next_states, progressions, rewards, dones = self.env.step(moves, self.step)
            
            self.memory.push(states, actions, next_states, rewards, dones)
            states = next_states
            
            if self.env.done:
                break

        self.rewards.append(rewards[0])
        self.progressions.append(progressions[0])
        if progressions[0] >= self.best_score:
            self.best_weights = self.dqn.model.get_weights()
            self.best_score = progressions[0]
        

    def epsilon_greedy_policy(self, states):
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.episode / EPS_DECAY)
        if np.random.rand() < self.epsilon:
            return [np.random.choice(np.arange(4), p=[4/6, 1/6, 1/6, 0])] # on favorise l'action up
        else:
            Q_values = self.dqn.model.predict(np.array(states), verbose=0)[0]
            return [Q_values.argmax()] # optimal action according to the DQN 
        
        
        



    def training_step(self):
        states, actions, next_states, rewards, dones = self.memory.sample()
        states = np.array(states).reshape(BATCH_SIZE, self.dqn.input_shape[0])
        
        # next_Q_values = self.dqn.model.predict(np.array(next_states), verbose=0)
        next_Q_values = self.target.predict(np.array(next_states), verbose=0)  # CHANGED
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = np.ones(len(dones)) - np.array(dones)
        target_Q_values = rewards + runs * DISCOUNT_FACTOR * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        
        mask = tf.one_hot(actions, self.dqn.output_shape)
        with tf.GradientTape() as tape:
            all_Q_values = self.dqn.model(np.array(states))
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.dqn.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dqn.model.trainable_variables))
        
        
        
        
        
        
        
    
if __name__=='__main__':
    
    algo = Trainer(warm_start=False)
    start = time.time()
    algo.train()
    print(f"Durée entrainement en {algo.episode} épisodes : {(time.time() - start)/60:.2f}min")
    
    
    if not algo.env.quit:
        n_train = len(os.listdir(WEIGHTS_PATH)) # nb de fichiers dans dossier weights
        with open(WEIGHTS_PATH / Path(f"{n_train}.weights"), "wb") as f:
            pickle.dump((algo.best_weights), f)
        
        plt.figure(figsize=(8, 4))
        plt.plot(algo.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Sum of rewards")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(IMAGES_PATH / Path(f"{n_train}_rewards_plot.png"), format="png", dpi=300)
        plt.show()
    
    







    
    




        



# Impossible de faire plusieurs actions en meme temps



