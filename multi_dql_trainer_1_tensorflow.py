import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers

from multi_dql_game_1 import Session
import library as lib           # ta lib inchangée

# ------------ hyper‑paramètres (inchangés) -------------
LR              = 0.001
GAMMA           = 0.95
SYNC_RATE       = 500
BATCH_SIZE      = 32
EPS_DECAY       = 1
EPS_DECAY_RATE  = 0.999 #pour allonger temps d'exploration
EPS_MIN         = 0.2
NUM_EPISODES    = 1400
MEMORY_LEN      = 400_000 #MEMORY_LEN = POPULATION_SIZE * 200 * steps * 2 #steps : episode conservés
PLOT_RATE       = 100
POPULATION_SIZE = 500
EPS_START = 1
# --------------------------------------------------------

rd.seed(0)
tf.random.set_seed(0)


# ================= Réseau DQN (Keras) ===================
class DQN(tf.keras.Model):
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
    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_LEN)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self):
        batch = rd.sample(self.buffer, BATCH_SIZE)
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
    def __init__(self, render):
        self.env = Session(nb_cars=POPULATION_SIZE, display=render)
        self.num_states  = len(self.env.observation_space)
        self.num_actions = len(self.env.action_space)

        # réseaux : policy + target
        self.policy_dqn = DQN(self.num_states, self.num_actions)
        self.target_dqn = DQN(self.num_states, self.num_actions)
        self.target_dqn.set_weights(self.policy_dqn.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.loss_fn   = tf.keras.losses.MeanSquaredError()

    # ------------ normalisation (version TF) ---------------
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
        plt.plot(rewards, label="Rewards")
        plt.plot(lib.moving_average(rewards), color='black', label='Window avg')
        plt.title("Rewards sum per episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(f'rewards_episode_{len(rewards)}_num_ep{NUM_EPISODES}_pop_size{POPULATION_SIZE}.png')

    # ------------------ optimisation -----------------------
    @tf.function
    def _train_step(self, states, actions, new_states, rewards, dones):
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones   = tf.cast(dones,   tf.float32)

        # cible Q
        q_next      = self.target_dqn(new_states)
        max_q_next  = tf.reduce_max(q_next, axis=1)
        target_qval = rewards + (1. - dones) * GAMMA * max_q_next

        with tf.GradientTape() as tape:
            q_vals = self.policy_dqn(states)                   # [B, num_actions]
            # idx    = tf.stack([tf.range(BATCH_SIZE), actions], axis=1)
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
        memory  = ReplayMemory()
        rewards_per_episode = []
        best_rewards = -float('inf')

        for episode in range(1, NUM_EPISODES):
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
                for i in range(POPULATION_SIZE):
                    memory.append((states[i], actions[i],
                                   new_states[i], rewards_list[i],
                                   float(terminated_list[i])))

                states     = new_states
                rewards   += sum(rewards_list)
                step      += 1
                terminated = all(terminated_list) or self.env.episode_done

            if self.env.quit: break

            rewards_per_episode.append(rewards)


            # apprentissage
            if len(memory) > BATCH_SIZE:
                s,a,ns,r,d = memory.sample()
                self._train_step(self.normalisation(s),a,self.normalisation(ns),r, d)

                #epsilon = max(epsilon - EPS_DECAY/NUM_EPISODES, EPS_MIN)

                #epsilon = max(EPS_MIN, epsilon * (EPS_DECAY_RATE ** episode))
                epsilon = max(EPS_MIN, EPS_START * (EPS_DECAY_RATE ** episode))
                # sync cible
                if episode % SYNC_RATE == 0:
                    self.target_dqn.set_weights(self.policy_dqn.get_weights())

            # save best
            if rewards > best_rewards:
                best_rewards = rewards
                self.policy_dqn.save_weights(filepath)

            if episode % PLOT_RATE == 0:
                self.plot_progress(rewards_per_episode)

            print(f'Episode {episode}, epsilon {epsilon:.2f}, sum_rewards {rewards:7.2f},'
                  f' memory {len(memory)}')
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
    agent = DQL(render=False)
    agent.train("weights_2_tf")
    agent.test("weights_2_tf")





# TODO
# print distance moyenne et max parcourue a chaque itération
