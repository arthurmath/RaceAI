import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import tensorflow as tf
from dqn_game import Session

from collections import deque
replay_buffer = deque(maxlen=2000)



SEED = 42
LR = 1e-2
N_STEPS = 200
BATCH_SIZE = 4 # 32
N_EPISODES = 20 #600
EPS_FACTOR = int(N_EPISODES * 5 / 6) 
DISCOUNT_FACTOR = 0.95
WEIGHTS_PATH = Path() / "results_dqn/weights"
IMAGES_PATH = Path() / "results_dqn/images"
VIDEOS_PATH = Path() / "results_dqn/videos"

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
            tf.keras.layers.Dense(self.output_shape)
        ])
        
        # self.previous_moves = [0, 0]
        
        
    def train(self):
        
        self.env = Session(display=True)
        rewards = [] 
        best_score = 0

        for episode in range(N_EPISODES):
            
            done = False
            state = self.env.reset()
            
            episode_reward = 0
            for step in range(N_STEPS + 20 * episode):
                epsilon = max(1 - episode / EPS_FACTOR, 0.01)
                state, reward, done = self.play_one_step(state, epsilon)
                episode_reward += reward
                if done:
                    break

            rewards.append(episode_reward)
            if episode_reward >= best_score:
                best_weights = self.model.get_weights()
                best_score = episode_reward

            if episode > 50:
                self.training_step()
                
            print(f"Episode: {episode + 1}, reward: {episode_reward:.2f}, done at step {step}")
            
        self.env.close()
        return rewards, best_weights
    
    
    
    def play_one_step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done = self.env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done


    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            # return np.random.randint(self.output_shape)  # random action
            return np.random.choice(np.arange(4), p=[3/6, 1/6, 1/6, 1/6]) # on favorise l'action 'avant'
        else:
            Q_values = self.model.predict(np.array(state)[np.newaxis, :], verbose=0)[0]
            return Q_values.argmax()  # optimal action according to the DQN


    def training_step(self):
        optimizer = tf.keras.optimizers.Nadam(learning_rate=LR)
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        states, actions, rewards, next_states, dones = self.sample_experiences()
        
        next_Q_values = self.model.predict(np.array(next_states))
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = np.ones(len(dones)) - np.array(dones)
        target_Q_values = rewards + runs * DISCOUNT_FACTOR * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        
        mask = tf.one_hot(actions, self.output_shape)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(np.array(states))
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  
        
        
    def sample_experiences(self):
        indices = np.random.randint(len(replay_buffer), size=BATCH_SIZE)
        batch = [replay_buffer[index] for index in indices]
        return [[experience[field_index] for experience in batch] for field_index in range(5)] 
        
        
        
        
        
    
def main():
    
    train = False
    dqn = DeepQN()
    n_train = len(os.listdir(WEIGHTS_PATH))
    
    if train:
        rewards, best_weights = dqn.train()
        
        with open(WEIGHTS_PATH / Path(f"{n_train}.weights"), "wb") as f:
            pickle.dump((best_weights), f)
        
        plt.figure(figsize=(8, 4))
        plt.plot(rewards)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of rewards", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(IMAGES_PATH / Path(f"{n_train}_rewards_plot.png"), format="png", dpi=300)
        plt.show()
        
    else:
        with open(WEIGHTS_PATH / Path(f"{n_train-1}.weights"), "rb") as f:
            weights = pickle.load(f)
        
        dqn.model.set_weights(weights)
        
        env = Session(display=True)
        state = env.reset()
        done = False
    
        while not done:
            moves = dqn.model.predict(np.array(state)[np.newaxis, :], verbose=0)[0].argmax()
            state, _, done = env.step(moves)
        env.close()





if __name__=='__main__':
    main()


    
    




        



# Impossible de faire plusieurs actions en meme temps


# next_state = self.normalize_state(next_state)
# next_state.append(self.previous_moves) # #######

# prendre le espilon exponentiel de gpt.py