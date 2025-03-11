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
N_EPISODES = 600
EPS_FACTOR = int(N_EPISODES * 5 / 6) 
DISCOUNT_FACTOR = 0.95
BATCH_SIZE = 32
WEIGHTS_PATH = Path() / "results_dqn/weights"
IMAGES_PATH = Path() / "results_dqn/images"
VIDEOS_PATH = Path() / "results_dqn/videos"

np.random.seed(SEED)
tf.random.set_seed(SEED)





class DeepQN:
    def __init__(self):
        
        self.env = Session()
        
        self.input_shape = [6]  # == env.observation_space.shape
        self.n_outputs = 4  # == env.action_space.n

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(32, activation="elu"),
            tf.keras.layers.Dense(self.n_outputs)
        ])
        
        self.fitness = 0
        self.nbMove = 0
        self.actions = ['U', 'D', 'L', 'R']
        self.previous_moves = [0, 0]
        
        
    def train(self):
        rewards = [] 
        best_score = 0

        for episode in range(N_EPISODES):
            state = self.env.reset()
            episode_reward = 0
            for step in range(N_STEPS):
                epsilon = max(1 - episode / EPS_FACTOR, 0.01)
                state, reward, done = self.play_one_step(state, epsilon)
                episode_reward += reward
                if done:
                    break

            rewards.append(episode_reward)
            if step >= best_score:
                best_weights = self.model.get_weights()
                best_score = step

            if episode > 50:
                self.training_step()
                
            print(f"Episode: {episode + 1}, reward: {episode_reward}, eps: {epsilon:.3f}")

        return rewards, best_weights
    
    
    
    def play_one_step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done = self.env.step(action)
        # next_state = self.normalize_state(next_state)
        # next_state.append(self.previous_moves) # #######
        replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done


    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)  # random action
        else:
            Q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
            return Q_values.argmax()  # optimal action according to the DQN


    def training_step(self):
        optimizer = tf.keras.optimizers.Nadam(learning_rate=LR)
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        states, actions, rewards, next_states, dones = self.sample_experiences()
        
        next_Q_values = self.model.predict(next_states, verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = 1.0 - (dones) 
        target_Q_values = rewards + runs * DISCOUNT_FACTOR * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  
        
        
    def sample_experiences(self):
        indices = np.random.randint(len(replay_buffer), size=BATCH_SIZE)
        batch = [replay_buffer[index] for index in indices]
        return [np.array([experience[field_index] for experience in batch]) for field_index in range(6)] 
        
        
        

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = matplotlib.animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=True, interval=50)  # interval in ms
    return anim

def show_one_episode(algo):
    frames = []
    state = algo.env.reset(SEED)
    for step in range(N_STEPS+1000):
        frames.append(algo.env.render())
        action = algo.model.predict(state[np.newaxis], verbose=0)[0].argmax()
        state, reward, done = algo.env.step(action)
        if done:
            print(f"Truncated at step {step}\n")
            break
    algo.env.close()
    return plot_animation(frames)
        
        
        
    
def main():
    
    train = False
    algo = DeepQN()
    n_train = len(os.listdir(WEIGHTS_PATH))
    
    if train:
        rewards, best_weights = algo.train()
        
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
        
        algo.model.set_weights(weights)

        anim = show_one_episode(algo)
        anim.save(VIDEOS_PATH / Path(f"{n_train-1}_cartpole_anim.mp4"), writer='ffmpeg', fps=30)
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







        

def reset_state(self):
    self.nbMove = 0