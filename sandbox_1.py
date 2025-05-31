import math
import time
import os, sys
import numpy as np
import random as rd
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)
from dqn_trainer import DeepQNetwork
import matplotlib.pyplot as plt

from collections import deque

#from dql_trainer_multi_1 import DQL  # Import de la classe DQL
from dql_game_multi_1 import Session  # Import de la classe Session
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



import torch
from torch import nn
import torch.nn.functional as F

import library as lib

rd.seed(42)




# vec = np.random.random((1, 5)) #ce
# #vec = np.random.random((16, 5))
# print(vec.shape)
# 
# dqn = DeepQNetwork()
# print(dqn.model.predict(vec).shape)


#
# # Example usage:
# env = MockEnv()
# processor = NormalisationProcessor(env)
#
# test_data = [
#     639.4267984578837, 25.010755222666937, 55.00586367382385, 80.3558657335762,
#     23.647121416401234, 338.34974371145563, -1, 32.57730448165427
# ]
#
# norm_1 = processor.normalisation_before(test_data)
# print(norm_1)
#
#
#
#
#
#
#
#
#Exemple d'utilisation
# test_states = generate_test_data(nb_cars=3)
# print(test_states)

# top = [5,2]
# POPULATION_SIZE = 3
# x = np.arange(1100)
# y = -11.5 + np.random.randn(1100) * (1 / (x + 1)**0.3) + np.log(x + 1) / 5
#
# def window_average(y, win):
#     return [sum(y[i : i+win]) / win for i in range(len(y) - win)]
#
# def moving_average(rewards_per_episode):
#     len_window = 100
#     moyenne_mobile = []
#     for i in range(len(rewards_per_episode)):
#         if i < len_window:
#             start_index = 0
#         else:
#             start_index = i - len_window + 1
#         window = rewards_per_episode[start_index: i + 1]
#         moyenne_mobile.append(sum(window) / len(window))
#     return moyenne_mobile
#
#
# win = 100
# plt.figure()
# plt.plot(y)
# plt.plot(x, moving_average(y), color='black')
# x = np.arange(int(win / 2), len(y)-int(win / 2))
# plt.plot(x, window_average(y, win))
# plt.title("Rewards sum per episode")
# plt.xlabel("Episode")
# plt.ylabel("Rewards")
# plt.show()
# plt.savefig(f'test_{len(top)}_Pop_{POPULATION_SIZE}.png')

#
# ------------ hyper‑paramètres (inchangés) -------------
LR              = 0.001
GAMMA           = 0.95
SYNC_RATE       = 500
BATCH_SIZE      = 64
EPS_DECAY       = 1
EPS_DECAY_RATE = 0.999 #pour allonger temps d'exploration
EPS_MIN         = 0.2
NUM_EPISODES    = 600
MEMORY_LEN      = 400_000 #MEMORY_LEN = POPULATION_SIZE * 200 * steps * 2 #steps : episode conservés
PLOT_RATE       = 100
POPULATION_SIZE = 50
EPS_START = 1
# --------------------------------------------------------

rd.seed(0)
tf.random.set_seed(0)


# ================= Réseau DQN (Keras) ===================
class DQN(keras.Model):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.fc1 = layers.Dense(32, activation="relu")
        self.fc2 = layers.Dense(32, activation="relu")
        self.out = layers.Dense(outputs, activation="relu")

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


# ================ Mémoire tampon identique ===============
class ReplayMemory:
    def __init__(self, maxlen :int, batch_size :int):
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self):
        batch = rd.sample(self.buffer, self.batch_size)
        states, actions, new_states, rewards, dones = map(list, zip(*batch))
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(new_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ======================= Agent DQL =======================
class DQL():
    def __init__(self,
                 render: bool,
                 lr: float = LR,
                 gamma: float = GAMMA,
                 batch_size: int = BATCH_SIZE,
                 epsilon_decay_rate: float = EPS_DECAY,
                 num_episodes: int = NUM_EPISODES,
                 population_size: int = POPULATION_SIZE):
        # on stocke chaque hyper‑paramètre sur self
        self.lr                  = lr
        self.gamma               = gamma
        self.batch_size          = batch_size
        self.epsilon_decay_rate  = epsilon_decay_rate
        self.num_episodes        = num_episodes
        self.population_size     = population_size


        self.env = Session(nb_cars=self.population_size, display=render)
        self.num_states  = len(self.env.observation_space)
        self.num_actions = len(self.env.action_space)

        # réseaux : policy + target
        self.policy_dqn = DQN(self.num_states, self.num_actions)
        self.target_dqn = DQN(self.num_states, self.num_actions)
        self.target_dqn.set_weights(self.policy_dqn.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=LR)
        self.loss_fn   = keras.losses.MeanSquaredError()

    # ------------ normalisation inchangée (version TF) ----
    # ------------ normalisation (version finale) ----------
    def normalisation(self, state):
        """
        • state = 1‑D  (num_states,)  -> Tensor (1, num_states)
        • state = 2‑D  (batch, num_states) -> Tensor (batch, num_states)
        """
        # assure un ndarray float32
        state  = np.asarray(state, dtype=np.float32)

        # mise à l'échelle feature par feature
        scaled = np.empty_like(state, dtype=np.float32)
        for i in range(self.num_states):
            a, b = self.env.observation_space[i]
            scaled[..., i] = lib.scale(state[..., i], a, b)

        tensor = tf.convert_to_tensor(scaled, dtype=tf.float32)

        # si l'entrée était un vecteur 1‑D, on ajoute la dim batch
        if len(tensor.shape) == 1:
            tensor = tf.expand_dims(tensor, axis=0)    # -> (1, num_states)

        return tensor


    # ------------ tracé  ----------------------------------
    def plot_progress(self, rewards):
        plt.figure()
        #plt.clf() #cett
        plt.plot(rewards, label="Rewards")
        plt.plot(lib.moving_average(rewards), color='black', label='Window avg')
        plt.title("Rewards sum per episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(f'rewards_episode_{len(rewards)}_num_ep{self.num_episodes}_pop_size{self.population_size}.png')
    # ------------------ optimisation -----------------------
    @tf.function
    def _train_step(self, states, actions, new_states, rewards, dones):
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones   = tf.cast(dones,   tf.float32)

        # cible Q
        q_next      = self.target_dqn(new_states)
        max_q_next  = tf.reduce_max(q_next, axis=1)
        target_qval = rewards + (1. - dones) * self.gamma * max_q_next

        with tf.GradientTape() as tape:
            q_vals = self.policy_dqn(states)                   # [B, num_actions]
            idx    = tf.stack([tf.range(self.batch_size), actions], axis=1)
            batch_sz = tf.shape(actions)[0]  # dynamique
            idx = tf.stack([tf.range(batch_sz), actions], axis=1)
            pred_q = tf.gather_nd(q_vals, idx)                 # [B]
            loss   = self.loss_fn(target_qval, pred_q)

        grads = tape.gradient(loss, self.policy_dqn.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.policy_dqn.trainable_weights)
        )

    # def action_distribution_strategy(self, episode):
    #     """Retourne la distribution de probabilité pour les actions selon l’épisode."""
    #     if episode < 500:
    #         return [0.6, 0.2, 0.2, 0.0]  # favorise fortement 'up'
    #     elif episode < 2000:
    #         return [0.4, 0.3, 0.3, 0.0]  # exploration latérale
    #     elif episode < 5000:
    #         return [0.3, 0.3, 0.3, 0.1]  # standard équilibré
    #     else:
    #         return [0.25, 0.25, 0.25, 0.25]  # exploration uniforme (fin d’apprentissage)

    # -------------------- Entraînement ---------------------
    def train(self, filepath: str) -> None:
        epsilon = 1.0
        memory  = ReplayMemory(maxlen=MEMORY_LEN, batch_size=BATCH_SIZE)
        rewards_per_episode = []
        best_rewards = -float('inf')

        for episode in range(1, self.num_episodes):
            states = self.env.reset(episode)
            rewards = 0
            step = 0
            terminated = False

            while not terminated and step < 500:
                actions = []
                #action_weights = self.action_distribution_strategy(episode)
                for state in states:
                    if rd.random() < epsilon:
                        #action = rd.choices(self.env.action_space,weights=action_weights)[0]
                        action = rd.choices(self.env.action_space, weights=[0.4,0.3,0.3,0])[0]
                    else:
                        qvals = self.policy_dqn(self.normalisation(state))
                        action = int(tf.argmax(qvals, axis=1)[0])
                    actions.append(action)

                new_states, rewards_list, terminated_list = self.env.step(actions)

                # stockage transitions
                for i in range(self.population_size):
                    memory.append((states[i], actions[i],
                                   new_states[i], rewards_list[i],
                                   float(terminated_list[i])))

                states     = new_states
                rewards   += sum(rewards_list)
                step      += 1
                terminated = all(terminated_list) or self.env.episode_done
            #print("step",step)

            if self.env.quit: break

            rewards_per_episode.append(rewards)


            # apprentissage
            if len(memory) > self.batch_size:
                s,a,ns,r,d = memory.sample()
                self._train_step(self.normalisation(s),a,self.normalisation(ns),r, d)

                #epsilon = max(epsilon - EPS_DECAY/NUM_EPISODES, EPS_MIN)

                #epsilon = max(EPS_MIN, epsilon * (EPS_DECAY_RATE ** episode))
                epsilon = max(EPS_MIN, EPS_START * (self.epsilon_decay_rate ** episode))
                # sync cible
                if episode % SYNC_RATE == 0:
                    self.target_dqn.set_weights(self.policy_dqn.get_weights())

            # save best
            if rewards > best_rewards:
                best_rewards = rewards
                self.policy_dqn.save_weights(filepath)

            if episode % PLOT_RATE == 0:
                self.plot_progress(rewards_per_episode)
            print(f"Episode {episode}")

            # print(f'Episode {episode}, epsilon {epsilon:.2f}, sum_rewards {rewards:7.2f},'
            #       f' memory {len(memory)}')
        self.rewards_per_episode = rewards_per_episode

        self.env.close()


    # ------------------- Inférence -------------------------
    def test(self, filepath):
        self.policy_dqn.load_weights(filepath)
        state = self.env.reset()
        terminated = False
        while not terminated:
            actions = [int(tf.argmax(
                self.policy_dqn(self.normalisation(s)), axis=1)[0]) for s in state]
            state, _, terminated_list = self.env.step(actions)
            terminated = any(terminated_list)
        self.env.close()


# ========================= main ============================
if __name__ == '__main__':
    agent = DQL(render=True)
    agent.train("weights_1_tf")
    #agent.test("weights_1_tf")


