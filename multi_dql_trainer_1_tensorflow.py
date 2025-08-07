import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from multi_dql_game_1 import Session
import library as lib 


# ------------ hyper‑paramètres (inchangés) -------------
LR              = 0.001
GAMMA           = 0.95 # facteur d'actualisation
SYNC_RATE       = 5 # nombre d'itérations avant de synchroniser le réseau cible basé sur les episodes
BATCH_SIZE      = 128
EPS_DECAY       = 1
EPS_DECAY_RATE  = 0.995 #pour allonger temps d'exploration
EPS_MIN         = 0.2
NUM_EPISODES    = 300
MEMORY_LEN      = 400_000 #MEMORY_LEN = POPULATION_SIZE * 200 * steps * 2 #steps : episode conservés
PLOT_RATE       = 5
POPULATION_SIZE = 50
EPS_START = 1
SEQUENCE_LEN = 5 # nombre de transitions à stocker dans la mémoire tampon
STEP_MAX = 10_000 # nombre maximum d'étapes par épisode

rd.seed(0)
tf.random.set_seed(0)


# ================= Réseau DQN (Keras) ===================
class DQN(tf.keras.Model):
    # def __init__(self, inputs, outputs):
    #     super().__init__()
    #     self.fc1 = tf.keras.layers.Dense(32, activation="relu")
    #     self.fc2 = tf.keras.layers.Dense(32, activation="relu")
    #     self.out = tf.keras.layers.Dense(outputs, activation= None)  # pas d'activation pour la couche de sortie pour les valeurs Q car on veut des valeurs réelles
    #
    #
    # def call(self, x):
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     return self.out(x)
    def __init__(self, sequence_len, num_features, outputs):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu")
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu")
        self.norm2 = tf.keras.layers.BatchNormalization() # normalisation des activations pour stabiliser l'entraînement
        self.global_pool = tf.keras.layers.GlobalMaxPooling1D() # pour réduire la dimensionnalité des séquences d'états
        self.dense = tf.keras.layers.Dense(64, activation="relu")  # couche dense pour la représentation des états
        self.out = tf.keras.layers.Dense(outputs, activation=None)  # couche de sortie pour les valeurs Q

    def call(self, x):
        #x.shape = (batch_size, SEQUENCE_LEN, num_features)  # x est un tenseur 3D
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)  # réduit la dimensionnalité des séquences d'états
        x = self.dense(x)
        return self.out(x)  # retourne les valeurs Q pour chaque action (batch_size, num_actions)


# ================ Mémoire tampon identique =============== 
class DualReplayMemory:
    """buffer_all : pour stocker toutes les transitions
    buffer_focus : pour stocker les transitions les plus performantes"""
    def __init__(self, MEMORY_LEN=MEMORY_LEN, MEMORY_FOCUS_LEN=MEMORY_LEN//10):
        self.buffer_all = deque(maxlen=MEMORY_LEN)
        self.buffer_focus = deque(maxlen=MEMORY_FOCUS_LEN)

    def append(self, transition, is_focus : bool = False):
        self.buffer_all.append(transition)
        if is_focus: #si transition est une transition focus, on l'ajoute à la mémoire focus
            self.buffer_focus.append(transition)
            
    def sample(self, BATCH_SIZE, k_fraction=0.2):
        k_focus = int(BATCH_SIZE * k_fraction)  # 20% des échantillons de la mémoire focus
        k_all = BATCH_SIZE - k_focus  # 80% des échantillons de la mémoire générale

        batch_focus = rd.sample(self.buffer_focus, min(k_focus, len(self.buffer_focus))) # pour éviter l'erreur si la mémoire focus est vide
        #si buffer_focus est vide, on prend des échantillons de la mémoire générale
        batch_all = rd.sample(self.buffer_all, min(k_all + (k_focus - len(batch_focus)), len(self.buffer_all)))
        batch = batch_focus + batch_all
        states, actions, new_states, rewards, dones = map(list, zip(*batch))
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(new_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer_all)




# ======================= Agent DQL =======================
class DQL():
    def __init__(self, render):
        self.env = Session(nb_cars=POPULATION_SIZE, display=render)
        self.num_states  = len(self.env.observation_space)
        self.num_actions = len(self.env.action_space)
        #
        # # réseaux : policy + target #pour la dernière action
        # self.policy_dqn = DQN(self.num_states, self.num_actions)
        # self.target_dqn = DQN(self.num_states, self.num_actions)
        # self.target_dqn.set_weights(self.policy_dqn.get_weights())
        #
        #================== gestion avec sequence de states =========================
        self.policy_dqn = DQN(SEQUENCE_LEN, self.num_states, self.num_actions)  # DQN avec convolutions
        self.target_dqn = DQN(SEQUENCE_LEN, self.num_states, self.num_actions)
        self.target_dqn.set_weights(self.policy_dqn.get_weights()) # copie des poids du réseau policy vers le réseau cible
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

    # ------------ Normalisation pour les séquences d'états --------------
    def normalisation_seq(self, seq_states):
        """
        • seq_states = 3‑D  (batch, SEQUENCE_LEN, num_states) -> Tensor (batch, SEQUENCE_LEN, num_states)
        """
        scaled = np.empty_like(seq_states, dtype=np.float32)
        for i in range(self.num_states):
            a, b = self.env.observation_space[i]
            scaled[..., i] = lib.scale(seq_states[..., i], a, b)
        return tf.convert_to_tensor(scaled, dtype=tf.float32)

    # ------------ tracé  ----------------------------------
    def plot_progress(self, rewards):
        plt.figure()
        plt.plot(rewards, label="Rewards")
        plt.plot(lib.moving_average(rewards), color='black', label='Window avg')
        plt.title("Rewards sum per episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(f'results_dqn/plots/rewards_episode_{len(rewards)}_num_ep{NUM_EPISODES}_pop_size{POPULATION_SIZE}.png')
        plt.show()

    def plot_distance(self, distance):
        plt.figure()
        plt.plot(distance, label="Distance max par épisode")
        plt.plot(lib.moving_average(distance), color='black', label='Moyenne mobile')
        plt.title("Distance max par épisode")
        plt.xlabel("Épisode")
        plt.ylabel("Distance (m)")
        plt.savefig(f'results_dqn/plots/distance_episode_{len(distance)}_num_ep{NUM_EPISODES}_pop_size{POPULATION_SIZE}.png')
        plt.show()

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


    # -------------------- Entraînement ---------------------
    def train(self, filepath: str) -> None:
        epsilon = 1.0
        global_step = 0  # compteur global pour suivre le nombre d'étapes
        memory = DualReplayMemory()  # mémoire tampon pour stocker les transitions
        rewards_per_episode = []
        distance_per_episode = [] #pour plot
        best_rewards = -float('inf')
        best_episode = 0  # Track which episode had the best performance
        best_distance = -float('inf')  # Track the best distance achieved

        for episode in range(1, NUM_EPISODES+1):
            states = self.env.reset(episode)
            state_history = [deque(maxlen=SEQUENCE_LEN) for _ in range(POPULATION_SIZE)] # pour stocker les états de chaque voiture
            max_dist_per_episode = [] # pour stocker la distance maximale parcourue par une voiture dans l'épisode
            #avant la première étape, on remplit la mémoire avec l'état initial
            for i, state in enumerate(states): #initialise les états pour chaque voiture au début d'un épisode, permet d'avoir une séquence d'états de longueur fixe
                for _ in range(SEQUENCE_LEN):
                    state_history[i].append(state)
            rewards = 0
            step = 0
            terminated = False
            max_progression_per_episode = 0 # pour stocker la distance maximale parcourue par une voiture dans l'épisode

            while not terminated and step < STEP_MAX: #parallèlisation des actions pour plus d'efficacité
                global_step += 1
                step += 1
                if rd.random() < epsilon:
                    actions = rd.choices(self.env.action_space, weights=[0.7, 0.15, 0.15, 0], k = POPULATION_SIZE)
                else:
                    # on construit un batch de séquences d'états pour chaque voiture
                    seq_batch = np.stack(state_history, axis=0) #shape (POPULATION_SIZE, SEQUENCE_LEN, num_states) (30,5,8)
                    seq_tensor = self.normalisation_seq(seq_batch)
                    q_vals = self.policy_dqn(seq_tensor)  # on prédit les valeurs Q pour chaque séquence d'états
                    actions = tf.argmax(q_vals, axis=1).numpy()  # on choisit l'action avec la valeur Q maximale pour chaque voiture

                new_states, rewards_list, terminated_list = self.env.step(actions)
                states = new_states  # on met à jour les états pour le prochain épisode
                current_progressions = [car.progression for car in self.env.car_list if car.alive]
                # print("current_progressions", current_progressions)
                max_dist_per_episode = max(max_progression_per_episode, max(current_progressions))  # on prend la distance maximale parcourue par une voiture dans l'épisode
                rewards += sum(rewards_list)
                terminated = all(terminated_list) or self.env.episode_done

                # Check if current distance performance is the best so far and save immediately
                if max_dist_per_episode > best_distance:
                    best_distance = max_dist_per_episode
                    self.policy_dqn.save_weights(filepath)
                    print(f"New best distance! Distance: {max_dist_per_episode:.2f}m - Weights saved")

                # current_progressions = [car.progression for car in self.env.car_list if car.alive]
                # max_progression = max(current_progressions)

                #================ Stocker une séquence d'états dans la mémoire tampon ================
                for i in range(POPULATION_SIZE):
                    is_focus = lib.is_focus_transition(self.env.car_list[i])  # vérifie si la voiture a une performance significative
                    seq_before = list(state_history[i]) #on crée une copie de l'historique avant de le tronquer pour la séquence d'après
                    state_history[i].append(new_states[i])  # on ajoute le nouvel état à l'historique de chaque voiture
                    seq_after = list(state_history[i])
                    #pour le double batch des meilleures performances
                    memory.append((seq_before, actions[i], seq_after, rewards_list[i], float(terminated_list[i])), is_focus)

                # pour double batch des meilleures performances
                if len(memory) > BATCH_SIZE:
                    s,a, ns, r, d = memory.sample(BATCH_SIZE, k_fraction=0.2)  # 20% des échantillons de la mémoire focus
                    self._train_step(self.normalisation_seq(s), a, self.normalisation_seq(ns), r, d)  # on entraîne le réseau policy
                    epsilon = max(EPS_MIN, EPS_START * (EPS_DECAY_RATE ** episode))  # epsilon décroît exponentiellement

                    # if global_step % SYNC_RATE == 0:
                    if episode % SYNC_RATE == 0:
                        self.target_dqn.set_weights(self.policy_dqn.get_weights())
                    states = new_states  # on met à jour les états pour le prochain épisode
            rewards_per_episode.append(rewards)
            distance_per_episode.append(max_dist_per_episode)  # pour tracer la distance maximale parcourue par une voiture dans l'épisode

            # Track which episode had the best rewards for logging purposes
            if rewards > best_rewards:
                best_episode = episode

            # Optional: Save checkpoint every N episodes for backup
            if episode % 20 == 0:
                checkpoint_path = f"{filepath}_episode_{episode}"
                self.policy_dqn.save_weights(checkpoint_path)

            if episode % PLOT_RATE == 0:
                self.plot_progress(rewards_per_episode)
                self.plot_distance(distance_per_episode)

            print(f'Episode {episode}, epsilon {epsilon:.2f}, sum_rewards {rewards:7.2f},'
                  f' memory {len(memory)}, max_distance: {max_dist_per_episode:.2f}m')
        
        print(f"Training completed. Best distance was {best_distance:.2f}m at episode {best_episode}")
        self.rewards_per_episode = rewards_per_episode

        self.env.close()


    # ------------------- Inférence -------------------------
    def test(self, filepath):
        self.policy_dqn.load_weights(filepath)
        state = self.env.reset()
        terminated = False
        state_history = [deque(maxlen=SEQUENCE_LEN) for _ in range(POPULATION_SIZE)]
        # on initialise l'historique des états pour chaque voiture
        for i, s in enumerate(state):
            for _ in range(SEQUENCE_LEN):
                state_history[i].append(s)
        terminated = False
        # on joue jusqu'à ce que toutes les voitures soient terminées
        while not terminated:
            # on crée un batch de séquences d'états pour chaque voiture
            seq_batch = np.stack(state_history, axis=0) # shape (POPULATION_SIZE, SEQUENCE_LEN, num_states)
            seq_tensor = self.normalisation_seq(seq_batch)
            # on prédit les actions pour chaque voiture
            qvals = self.policy_dqn(seq_tensor)
            actions = tf.argmax(qvals, axis=1).numpy().tolist() # on convertit en liste d'
            # on exécute les actions
            new_state, _, terminated_list = self.env.step(actions)
            # on met à jour l'historique des états pour chaque voiture
            terminated = any(terminated_list)
            for i in range(POPULATION_SIZE):
                state_history[i].append(new_state[i]) # on ajoute le nouvel état à l'historique de chaque voiture

        self.env.close()


# ========================= main ============================
if __name__ == '__main__':
    path = "weights/weights_2_tf"
    agent = DQL(render=True)
    agent.train(path)
    agent.test(path)
    plt.show()




# TODO
# print distance moyenne et max parcourue a chaque itération





























# UNUSED



#     terminated = all(terminated_list) or self.env.episode_done

#     if terminated:
#         break

# save best - REMOVED: Now saving immediately when performance improves


#calcul une action a la fois, un peu lent
# while not terminated and step < 2000:
#     actions = []
#     #action_weights = self.action_distribution_strategy(episode)
#     for state in states:
#         if rd.random() < epsilon:
#             #action = rd.choices(self.env.action_space,weights=action_weights)[0]
#             action = rd.choices(self.env.action_space, weights=[0.4,0.3,0.3,0])[0]
#         else:
#             qvals = self.policy_dqn(self.normalisation(state))
#             action = int(tf.argmax(qvals, axis=1)[0])
#         actions.append(action)



# apprentissage avec un seul replay memory
#     if len(memory) > BATCH_SIZE:
#         s,a,ns,r,d = memory.sample()
#         # self._train_step(self.normalisation(s),a,self.normalisation(ns),r, d) #pour un seul état
#         self._train_step(self.normalisation_seq(s),a,self.normalisation_seq(ns),r, d) #pour une séquence d'états
#         #epsilon = max(EPS_MIN, epsilon * (EPS_DECAY_RATE ** episode))
#         epsilon = max(EPS_MIN, EPS_START * (EPS_DECAY_RATE ** episode))
#         # epsilon = max(EPS_MIN, 1-global_step*(1-EPS_MIN)/STEP_MAX)  # epsilon décroît linéairement
#         # sync cible




#============= Ancienne version de test (une action à la fois) ================
# while not terminated:
#     actions = [int(tf.argmax(
#         self.policy_dqn(self.normalisation(s)), axis=1)[0]) for s in state]
#     state, _, terminated_list = self.env.step(actions)
#     terminated = any(terminated_list)





# ================ Mémoire tampon pour stocker dernières n transitions ===============
class ReplayMemory:
    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_LEN)

    def append(self, transition):
        # transition = (state_before, action, state_after, reward, done)
        #transition = (seq_before, action, seq_after, reward, done)
        self.buffer.append(transition)

    def sample(self):
        batch = rd.sample(self.buffer, BATCH_SIZE)
        seq_states, actions, seq_new_states, rewards, dones = map(list, zip(*batch))
        return (
            np.array(seq_states, dtype=np.float32),  # (BATCH, SEQUENCE_LEN, num_features)
            np.array(actions, dtype=np.int32),  # (BATCH,)
            np.array(seq_new_states, dtype=np.float32),  # (BATCH, SEQUENCE_LEN, num_features)
            np.array(rewards, dtype=np.float32),  # (BATCH,)
            np.array(dones, dtype=np.float32),  # (BATCH,)
        )

    def __len__(self):
        return len(self.buffer)
