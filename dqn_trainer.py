import os
import time
import pickle
import numpy as np
import random as rd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from dqn_game import Session

from collections import deque
replay_buffer = deque(maxlen=2000)



SEED = 42
LR = 1e-2
N_STEPS = 200
BATCH_SIZE = 16 # 32
N_EPISODES = 100 
EPS_FACTOR = int(N_EPISODES * 5 / 6) 
DISCOUNT_FACTOR = 0.95
WEIGHTS_PATH = Path() / "results_dqn/weights"
IMAGES_PATH = Path() / "results_dqn/images"

rd.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)






class DeepQN:
    def __init__(self):
        
        self.input_shape = [6]  # == env.observation_space.shape
        self.output_shape = 4  # == env.action_space.n

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(self.input_shape),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(self.output_shape)
        ])
        
        self.target = tf.keras.models.clone_model(self.model)  # CHANGED
        self.target.set_weights(self.model.get_weights())  # CHANGED
        
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=LR)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        # self.previous_moves = [0, 0]
        
        
    def train(self):
        
        self.env = Session(display=True)
        rewards = [] 
        best_score = 0

        for episode in range(N_EPISODES):
            
            state, done = self.env.reset()
            
            episode_reward = 0
            for step in range(N_STEPS + 10 * episode):
                epsilon = max(1 - episode / EPS_FACTOR, 0.01)
                state, reward, done = self.play_one_step(state, epsilon)
                episode_reward += reward
                # if done:
                #     break

            rewards.append(episode_reward)
            if episode_reward >= best_score:
                best_weights = self.model.get_weights()
                best_score = episode_reward

            if episode > int(BATCH_SIZE * 1.5):
                self.training_step()
                if episode % 50 == 0:                                  # CHANGED
                    self.target.set_weights(self.model.get_weights())  # CHANGED
                
            print(f"Episode: {episode + 1}, reward: {episode_reward:.2f}, done at step {step}, nb collisions: {self.env.car.nbCollisions}")
            
        self.env.close()
        return rewards, best_weights

    
    
    
    def play_one_step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done = self.env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done


    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            # return np.random.randint(self.output_shape)  # random action
            return np.random.choice(np.arange(4), p=[3/6, 1/6, 1/6, 1/6]) # on favorise l'action 'avant'
        else:
            Q_values = self.model.predict(np.array(state)[np.newaxis, :], verbose=0)[0]
            return Q_values.argmax()  # optimal action according to the DQN


    def training_step(self):
        
        states, actions, rewards, next_states, dones = self.sample_experiences()
        
        # next_Q_values = self.model.predict(np.array(next_states), verbose=0)
        next_Q_values = self.target.predict(np.array(next_states), verbose=0)  # CHANGED
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = np.ones(len(dones)) - np.array(dones)
        target_Q_values = rewards + runs * DISCOUNT_FACTOR * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        
        mask = tf.one_hot(actions, self.output_shape)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(np.array(states))
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  
        
        
    def sample_experiences(self):
        indices = np.random.randint(len(replay_buffer), size=BATCH_SIZE)
        batch = [replay_buffer[index] for index in indices]
        return [[experience[field_index] for experience in batch] for field_index in range(5)] 
        
        
        
        
        
    
def main():
    
    train = False
    dqn = DeepQN()
    n_train = len(os.listdir(WEIGHTS_PATH)) # nb de fichiers dans dossier weights
    
    if train:
        with open(WEIGHTS_PATH / Path(f"colab1.weights"), "rb") as f:
            weights = pickle.load(f) # warm start
            dqn.model.set_weights(weights)
        
        start = time.time()
        rewards, best_weights = dqn.train()
        print(f"Durée entrainement avec {N_EPISODES} épisodes : {(time.time() - start)/60}min")
        
        with open(WEIGHTS_PATH / Path(f"{n_train}.weights"), "wb") as f:
            pickle.dump((best_weights), f)
        
        plt.figure(figsize=(8, 4))
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Sum of rewards")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(IMAGES_PATH / Path(f"{n_train}_rewards_plot.png"), format="png", dpi=300)
        plt.show()
        
    else:
        with open(WEIGHTS_PATH / Path(f"colab3.weights"), "rb") as f:
            weights = pickle.load(f)
        
        dqn.model.set_weights(weights)
        
        env = Session(display=True)
        state, done = env.reset()
        done = False

        print("start")
        while not done:
            moves = dqn.model.predict(np.array(state)[np.newaxis, :], verbose=0)[0].argmax()
            state, _, done = env.step(moves)
            done = False
        env.close()





if __name__=='__main__':
    main()


    
    




        



# Impossible de faire plusieurs actions en meme temps


# next_state = self.normalize_state(next_state)
# next_state.append(self.previous_moves) # #######

# prendre le espilon exponentiel de gpt.py


