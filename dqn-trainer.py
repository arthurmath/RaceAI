from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gymnasium as gym
from gameV2 import Session

from collections import deque
replay_buffer = deque(maxlen=2000)

np.random.seed(42)
tf.random.set_seed(42)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    IMAGES_PATH = Path() / "images" / "rl"
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)





class DeepQ:
    def __init__(self):
        # self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        # self.env.reset(seed=42)  
        
        self.game = Session()
        
        self.input_shape = [6]  # == env.observation_space.shape
        self.n_outputs = 4  # == env.action_space.n

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="elu", input_shape=self.input_shape),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(self.n_outputs)
        ])
        
        self.batch_size = 32
        self.fitness = 0
        self.nbMove = 0
        self.actions = ['U', 'D', 'L', 'R']
        self.previous_moves = [0, 0]
        
        
    def train(self):
        rewards = [] 
        best_score = 0

        for episode in range(600):
            obs, info = self.game.reset()    
            for step in range(200):
                epsilon = max(1 - episode / 500, 0.01)
                obs, reward, done, truncated, info = self.play_one_step(self.game, obs, epsilon)
                if done or truncated:
                    break

            # extra code – displays debug info, stores data for the next figure, and
            #              keeps track of the best model weights so far
            print(f"\rEpisode: {episode + 1}, Steps: {step + 1}, eps: {epsilon:.3f}", end="")
            rewards.append(step) # ???
            if step >= best_score:
                best_weights = self.model.get_weights()
                best_score = step

            if episode > 50:
                self.training_step()

        self.model.set_weights(best_weights)  # restores the best model weights
        
        return rewards
    
    
    
    def play_one_step(self, game, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, truncated, info = game.step(action)
        next_state = self.normalize_state(next_state)
        next_state.append(self.previous_moves) # #######
        replay_buffer.append((state, action, reward, next_state, done, truncated))
        return next_state, reward, done, truncated, info


    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)  # random action
        else:
            Q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
            return Q_values.argmax()  # optimal action according to the DQN
        
        
    def normalize_state(self, state):
        # Il faut que les entrées soient dans [-1, 1] pour converger
        list_ranges = [[0, 1200], [0, 900], [-10, 10], [0, 360], [0, 1], [0, 500], [0, 100], [0, 3], [0, 3]]
        for idx, ranges in enumerate(list_ranges):
            state[idx] = self.scale(state[idx], *ranges)
        return state

    
    def scale(self, x, a, b):
        """Transforme la valeur x initialement comprise dans l'intervalle [a, b]
            en une valeur comprise dans l'intervalle [-1, 1]."""
        return 2 * (x - a) / (b - a) - 1


    def training_step(self):
        discount_factor = 0.95
        optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
        loss_fn = tf.keras.losses.mean_squared_error
        
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones, truncateds = experiences
        
        next_Q_values = self.model.predict(next_states, verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = 1.0 - (dones | truncateds)  # episode is not done or truncated
        target_Q_values = rewards + runs * discount_factor * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  
        
        
    def sample_experiences(self):
        indices = np.random.randint(len(replay_buffer), size=self.batch_size)
        batch = [replay_buffer[index] for index in indices]
        return [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(6)
        ]  # [states, actions, rewards, next_states, dones, truncateds] 
        
        
        


        
        
        
    
def main():
    
    algo = DeepQ()
    rewards = algo.train()
    
    
    # extra code
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.grid(True)

    save_fig("dqn_rewards_plot")

    plt.show()



if __name__=='__main__':
    main()


    
    




        

def choose_next_move(self, car):
    """ Choose a new move based on its state.
    Return the movement choice of the snake (tuple) """

        
    # Actions décidées par le réseau de neurones
    movesValues = self.adn.neural_network_forward(vision) 
    movesValues = movesValues.tolist()[0]  # listes de flottants dans [0, 1]
    
    # Choix des meilleures actions (celles avec une valeur > 0.7)
    choices = []
    for idx, x in enumerate(movesValues):
        if x > 0.6: # arbitraire
            choices.append(idx) # listes d'entiers dans [1, 4]

    self.previous_moves.extend(choices)
    while len(self.previous_moves) > 2:
        self.previous_moves.pop(0) # on ne garde que les 2 derniers moves
        
    self.nbMove += len(choices)
    
    self.moves = [self.actions[choice] for choice in choices]
        
    return self.moves





def compute_fitness(self, car):
    self.fitness = car.progression ** 2 / car.nbCollisions if car.nbCollisions else car.progression ** 2
    return self.fitness

        

def reset_state(self):
    self.nbMove = 0