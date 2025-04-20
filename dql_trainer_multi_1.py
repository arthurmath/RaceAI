import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from dql_game_multi_1 import Session
import library as lib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#faire en sorte d'enregistrer les paramètres du modèle quand on le lance pour avoir une DB des paramètres


LR = 0.01                    # learning rate
GAMMA = 0.95                 # discount rate
SYNC_RATE = 500              # number of steps the agent takes before syncing target with policy network
BATCH_SIZE = 32              # size of the data set sampled from the replay memory / utilisé pour traiter plusieurs états en même temps
EPS_DECAY = 6                # epsilon decay rate
EPS_MIN = 0.1                # minimum value of epsilon
NUM_EPISODES = 1000        # number of episodes
MEMORY_LEN = 20_000          # size of the buffer
PLOT_RATE = 100              # when to plot rewards
POPULATION_SIZE = 1000

rd.seed(0)




class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 32)   # fully-connected layer
        self.fc2 = nn.Linear(32, 32)       # fully-connected layer
        self.out = nn.Linear(32, outputs)  # ouptut layer

    def forward(self, x):
        x = F.relu(self.fc1(x))     # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        return x


class ReplayMemory(): #implémente une mémoire tampon pour stocker les transitions
    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_LEN)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self):
        return rd.sample(self.buffer, BATCH_SIZE) #

    def __len__(self):
        return len(self.buffer)


class DQL():
    def __init__(self, render):
        self.env = Session(nb_cars=POPULATION_SIZE, display=render)
        self.num_states = len(self.env.observation_space) # 4: x, y, speed, angle
        self.num_actions = len(self.env.action_space) # 4: up, down, left, right

#resumé train: entraine le modèle DQN entraine un réseau de neurones pour prédire les valeurs Q, on utilise les experiences collectées
    #stratégie epsilon-greedy pour choisir les actions, on utilise une mémoire tampon pour stocker les expériences
    #et on optimise le modèle en utilisant la perte MSE entre les valeurs Q prédites et les valeurs cibles
    def train(self, filepath: str) -> None:

        epsilon = 1 # 100% random actions
        memory = ReplayMemory()

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(self.num_states, self.num_actions)
        target_dqn = DQN(self.num_states, self.num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict()) # Copy weights/biases from policy to the target

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=LR)  #optimimiseur Adam
        self.loss_fn = nn.MSELoss() #fonction de perte

        # List to keep track of rewards and epsilon.
        rewards_per_episode = []

        # Track number of steps taken. Used for syncing policy => target network.
        best_rewards = - float('inf')

        for episode in range(1, NUM_EPISODES):
            states = self.env.reset(episode) #reset l'environnement et renvoie l'état initial
            terminated = False * POPULATION_SIZE  # True when agent falls in hole or reached goal
            rewards = 0
            step = 0
## code modifié
            # Agent navigates map until it collides a wall/reaches goal (terminated), or has taken 200 actions (truncated).
            while not terminated and step < 200: #c'est ici que l'on va boucler pour les N voitures
                actions: list = []
                for i, state in enumerate(states):
                    # Select action based on epsilon-greedy
                    if rd.random() < epsilon:
                        action = rd.choices(self.env.action_space, weights=[0.3, 0.3, 0.3, 0.1])[0]
                    else:
                        with torch.no_grad():
                            action = policy_dqn(self.normalisation(state)).argmax().item()
                    actions.append(action)
                new_states, rewards_list, terminated_list = self.env.step(actions) #on exécute l'action et on récupère le nouvel état
                # , la récompense et si l'agent est mort
                # Vérification des tailles des listes avant d'ajouter les transitions dans la mémoire
                assert len(states) == POPULATION_SIZE, f"states a une taille incorrecte : {len(states)}"
                assert len(actions) == POPULATION_SIZE, f"actions a une taille incorrecte : {len(actions)}"
                assert len(new_states) == POPULATION_SIZE, f"new_states a une taille incorrecte : {len(new_states)}"
                assert len(
                    rewards_list) == POPULATION_SIZE, f"rewards_list a une taille incorrecte : {len(rewards_list)}"
                assert len(
                    terminated_list) == POPULATION_SIZE, f"terminated_list a une taille incorrecte : {len(terminated_list)}"

                # Ajout des transitions dans la mémoire

                # Save experience into memory
                for i in range(POPULATION_SIZE):
                    memory.append((states[i], actions[i], new_states[i], rewards_list[i], terminated_list[i])) #on stocke la transition dans la mémoire
                #print((state, action, new_state, reward, terminated))

                states = new_states

                rewards += sum(rewards_list)
                step += 1

                if self.env.episode_done:
                    break

            if self.env.quit:
                break

            rewards_per_episode.append(rewards)

            # Graph training progress
            if episode % PLOT_RATE == 0:
                self.plot_progress(rewards_per_episode)

            if rewards > best_rewards:
                best_rewards = rewards
                torch.save(policy_dqn.state_dict(), filepath)

            # Check if enough experience has been collected
            if len(memory) > BATCH_SIZE: #echantillonnage d'un mini lot de transitions
                mini_batch = memory.sample()
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - EPS_DECAY / NUM_EPISODES, EPS_MIN)

                # Copy policy network weights to target network
                if episode % SYNC_RATE == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            print(f'Episode {episode}, epsilon {epsilon:.2f}, reward: {rewards:>7.2f}, memory: {len(memory)}')

        self.env.close()


    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch: #liste des transitions tuple, dimensions (BATCH_SIZE, 5)

            if terminated:
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward]).squeeze(0)
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + GAMMA * target_dqn(self.normalisation(new_state)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.normalisation(state)) #current_q est un tenseur de dimension (1, 4), on passe
            #l'état state dans le réseau de neurones en normalisation
            current_q_list.append(current_q.squeeze(0)) #current_q_list est une liste de tenseurs de dimension (1, 4) #######ici modif .squeeze(0) pour enlever la dimension 0
            #on met a jour le tenseur current_q_list avec l'état state

            # Get the target set of Q values
            target_q = target_dqn(self.normalisation(state)).detach() #on met a jour le tenseur target_q avec l'état state
            # Adjust the specific action to the target that was just calculated
            target_q = target_q.squeeze(0) #on enlève la dimension 0 du tenseur target_q
            target_q[action] = target #on met a jour le tenseur target_q avec l'action action
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        #current_q_list = current_q_list.squeeze(1) #on enlève la dimension 1 du tenseur current_q_list
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def normalisation(self, state): #préférable d'utiliser cette fonction car gère plusieurs états
        """ Scales nn inputs to [-1, 1] for better convergence """
        if isinstance(state[0], list): #on prend
            norm_states = []
            for s in state:
                norm_s = [lib.scale(s[i], *self.env.observation_space[i]) for i in range(len(s))]
                norm_states.append(norm_s)
            return torch.FloatTensor(norm_states)
        else:
            state = [lib.scale(state[i], *self.env.observation_space[i]) for i in range(len(state))]
            return torch.FloatTensor(state).unsqueeze(0) #pour modifier le tenseur avec la forme [1,n]


    def plot_progress(self, rewards):
        plt.figure()
        plt.plot(rewards, label='Rewards')
        plt.plot(lib.moving_average(rewards), color='black', label='Window average')
        plt.title("Rewards sum per episode")
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.savefig(f'rewards_episode_{len(rewards)}.png')



    # Run the environment with the learned policy
    def test(self, filepath):
        # Load learned policy
        policy_dqn = DQN(self.num_states, self.num_actions)
        policy_dqn.load_state_dict(torch.load(filepath))
        policy_dqn.eval()    # switch model to evaluation mode

        state = self.env.reset()
        terminated = False      # True when agent falls in hole or reached goal

        while not terminated:
            with torch.no_grad():
                action = policy_dqn(self.normalisation(state)).argmax().item()

            state, reward, terminated = self.env.step(action)

        self.env.close()

'''
# ----------------------- Hyper‑params -----------------------
LR            = 0.01       # learning rate
GAMMA         = 0.95       # discount rate
SYNC_RATE     = 500        # steps avant sync target <- policy
BATCH_SIZE    = 32         # taille du mini‑batch
EPS_DECAY     = 6          # décrément epsilon par épisode
EPS_MIN       = 0.1        # epsilon minimal
NUM_EPISODES  = 10000      # nombre d’épisodes
MEMORY_LEN    = 20000      # capacité mémoire tampon
PLOT_RATE     = 100        # fréquence de traçage
POPULATION_SIZE = 1000     # nb de voitures
# -----------------------------------------------------------
'''
rd.seed(0)
tf.random.set_seed(0)

#
# # ===================== 1. Réseau DQN =====================
# class DQN(keras.Model):
#     def __init__(self, inputs, outputs):
#         super(DQN, self).__init__()
#         self.fc1 = layers.Dense(32, activation="relu")
#         self.fc2 = layers.Dense(32, activation="relu")
#         self.out = layers.Dense(outputs, activation="relu")
#
#     def call(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return self.out(x)
#
#
# # ===================== 2. ReplayMemory ======================
# class ReplayMemory:
#     def __init__(self):
#         self.buffer = deque(maxlen=MEMORY_LEN)
#
#     def append(self, transition):
#         self.buffer.append(transition)
#
#     def sample(self):
#         batch = rd.sample(self.buffer, BATCH_SIZE)
#         states, actions, new_states, rewards, dones = map(list, zip(*batch))
#         return (
#             np.array(states, dtype=np.float32),
#             np.array(actions, dtype=np.int32),
#             np.array(new_states, dtype=np.float32),
#             np.array(rewards, dtype=np.float32),
#             np.array(dones, dtype=np.float32),
#         )
#
#     def __len__(self):
#         return len(self.buffer)
#
#
# # ===================== 3. Agent DQL =========================
# class DQL:
#     def __init__(self, render=False):
#         self.env           = Session(nb_cars=POPULATION_SIZE, display=render)
#         self.num_states    = len(self.env.observation_space)
#         self.num_actions   = len(self.env.action_space)
#
#         # networks
#         self.policy_dqn = DQN(self.num_states, self.num_actions)
#         self.target_dqn = DQN(self.num_states, self.num_actions)
#         # initial sync
#         self.target_dqn.set_weights(self.policy_dqn.get_weights())
#
#         # optimizer & loss
#         self.optimizer = keras.optimizers.Adam(learning_rate=LR)
#         self.loss_fn   = keras.losses.MeanSquaredError()
#
#     def normalisation(self, state):
#         """Scale inputs to [-1, 1], retourne un tensor TF"""
#         if isinstance(state[0], list):
#             norm_states = []
#             for s in state:
#                 norm_s = [
#                     lib.scale(s[i], *self.env.observation_space[i])
#                     for i in range(min(len(s)), len(self.env.observation_space))
#                 ]
#                 norm_states.append(norm_s)
#             return tf.convert_to_tensor(norm_states, dtype=tf.float32)
#         else:
#             flat = [
#                 lib.scale(state[i], *self.env.observation_space[i])
#                 for i in range(min(len(state)), )
#             ]
#             return tf.expand_dims(tf.convert_to_tensor(flat, dtype=tf.float32), axis=0)
#
#     def plot_progress(self, rewards_per_episode):
#         plt.figure(1)
#         plt.clf()
#         plt.title("Rewards per Episode")
#         plt.plot(rewards_per_episode, label="sum")
#         plt.plot(lib.moving_average(rewards_per_episode),
#                  label="avg", color="black")
#         plt.xlabel("Episode")
#         plt.ylabel("Total Reward")
#         plt.legend()
#         plt.pause(0.01)
#
#     def train(self, filepath: str) -> None:
#         epsilon = 1.0
#         memory  = ReplayMemory()
#         rewards_per_episode = []
#         best_rewards = -float("inf")
#
#         for episode in range(1, NUM_EPISODES + 1):
#             states = self.env.reset(episode)
#             total_reward = 0.0
#             step = 0
#             done_any = False
#
#             while (not done_any) and step < 200:
#                 actions = []
#                 for st in states:
#                     if rd.random() < epsilon:
#                         a = rd.choices(self.env.action_space,
#                                        weights=[0.3,0.3,0.3,0.1])[0]
#                     else:
#                         q = self.policy_dqn(self.normalisation(st))
#                         a = int(tf.argmax(q, axis=1)[0])
#                     actions.append(a)
#
#                 new_states, rewards_list, terminated_list = self.env.step(actions)
#
#                 # stocke chaque transition
#                 assert len(states) == POPULATION_SIZE
#                 for i in range(POPULATION_SIZE):
#                     memory.append((
#                         states[i],
#                         actions[i],
#                         new_states[i],
#                         rewards_list[i],
#                         float(terminated_list[i])
#                     ))
#
#                 states = new_states
#                 total_reward += sum(rewards_list)
#                 step += 1
#                 done_any = any(terminated_list) or self.env.episode_done
#
#             # fin épisode ou quit
#             if self.env.quit:
#                 break
#
#             rewards_per_episode.append(total_reward)
#
#             # apprentissage si mémoire suffisante
#             if len(memory) > BATCH_SIZE:
#                 mb = memory.sample()
#                 self.optimize(mb, self.policy_dqn, self.target_dqn)
#
#                 # sync target
#                 if episode % SYNC_RATE == 0:
#                     self.target_dqn.set_weights(
#                         self.policy_dqn.get_weights()
#                     )
#
#             # epsilon‑greedy decay
#             epsilon = max(epsilon - EPS_DECAY / NUM_EPISODES, EPS_MIN)
#
#             # save best
#             if total_reward > best_rewards:
#                 best_rewards = total_reward
#                 self.policy_dqn.save_weights(filepath)
#
#             # trace métriques
#             if episode % PLOT_RATE == 0:
#                 self.plot_progress(rewards_per_episode)
#             print(f"Epi {episode} ε={epsilon:.3f}  R={total_reward:.1f}")
#
#         self.env.close()
#
#     def optimize(self, mini_batch, policy_dqn, target_dqn):
#         # dépaqueter batch
#         s_batch, a_batch, ns_batch, r_batch, d_batch = mini_batch
#         # normalisation
#         s_batch_tf  = self.normalisation(list(s_batch))
#         ns_batch_tf = self.normalisation(list(ns_batch))
#
#         # calcul target Q
#         next_q    = target_dqn(ns_batch_tf)              # [B, num_actions]
#         max_next  = tf.reduce_max(next_q, axis=1)        # [B]
#         target_q  = r_batch + (1.0 - d_batch) * GAMMA * max_next
#
#         with tf.GradientTape() as tape:
#             q_values = policy_dqn(s_batch_tf)            # [B, num_actions]
#             idx       = tf.stack([tf.range(BATCH_SIZE), a_batch], axis=1)
#             pred_q    = tf.gather_nd(q_values, idx)      # [B]
#             loss      = self.loss_fn(target_q, pred_q)
#
#         grads = tape.gradient(loss, policy_dqn.trainable_weights)
#         self.optimizer.apply_gradients(
#             zip(grads, policy_dqn.trainable_weights)
#         )
#
#     def test(self, filepath: str) -> None:
#         # charge best poids
#         self.policy_dqn.load_weights(filepath)
#         states = self.env.reset()
#         done_any = False
#
#         while not done_any:
#             actions = []
#             for st in states:
#                 q = self.policy_dqn(self.normalisation(st))
#                 actions.append(int(tf.argmax(q, axis=1)[0]))
#             states, _, dones = self.env.step(actions)
#             done_any = any(dones)
#
#         self.env.close()
#
#
# if __name__ == "__main__":
#     agent = DQL(render=True)
#     agent.train("weights_tf")


###Tensorflow


if __name__ == '__main__':
   filepath = "weights.pt"

   mountaincar = DQL(render=True)
   mountaincar.train(filepath)
    #mountaincar.test(filepath)
    
    
    
    








# num_divisions = 30
# # Divide position and velocity into segments
# self.x_space = np.linspace(self.env.observation_space[0][0], self.env.observation_space[0][1], self.num_divisions) 
# self.y_space = np.linspace(self.env.observation_space[1][0], self.env.observation_space[1][1], self.num_divisions)
# self.speed_space = np.linspace(self.env.observation_space[2][0], self.env.observation_space[2][1], self.num_divisions)
# self.angle_space = np.linspace(self.env.observation_space[3][0], self.env.observation_space[3][1], self.num_divisions)

# def state_to_dqn_input(self, state) -> torch.Tensor:
#     ''' Converts a state (position, velocity) to tensor representation.
#     Example: Input = (0.3, -0.03) -> Return = tensor([16, 6]) '''
#     state_x = np.digitize(state[0], self.x_space)
#     state_y = np.digitize(state[1], self.y_space)
#     state_s = np.digitize(state[2], self.speed_space)
#     state_a = np.digitize(state[3], self.angle_space)
#     return torch.FloatTensor([state_x, state_y, state_s, state_a])