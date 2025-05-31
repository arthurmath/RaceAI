import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from dql_game_multi import Session
import library as lib
#from tests.test_generation import POPULATION

LR = 0.01                    # learning rate
GAMMA = 0.95                 # discount rate
SYNC_RATE = 500              # number of steps the agent takes before syncing target with policy network
BATCH_SIZE = 32              # size of the data set sampled from the replay memory
EPS_DECAY = 4                # epsilon decay rate 
EPS_MIN = 0.1                # minimum value of epsilon
NUM_EPISODES = 10_000        # number of episodes
MEMORY_LEN = 10_000          # size of the buffer
PLOT_RATE = 100              # when to plot rewards
POPULATION =2

rd.seed(0)




class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 20)   # fully-connected layer
        self.fc2 = nn.Linear(20, 20)       # fully-connected layer
        self.out = nn.Linear(20, outputs)  # ouptut layer

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0) #converti 2D en 1D
        x = F.relu(self.fc1(x))     # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        return x


class ReplayMemory():
    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_LEN)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self):
        return rd.sample(self.buffer, BATCH_SIZE)

    def __len__(self):
        return len(self.buffer)


class DQL(): 
    def __init__(self, render):
        self.env = Session(nb_cars=POPULATION, display=render)
        self.num_states = len(self.env.observation_space) # 4: x, y, speed, angle
        self.num_actions = len(self.env.action_space) # 4: up, down, left, right


    def train(self, filepath):
        epsilon = 1 # 100% random actions
        memory = ReplayMemory()

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(self.num_states, self.num_actions)
        target_dqn = DQN(self.num_states, self.num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict()) # Copy weights/biases from policy to the target
        
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        # List to keep track of rewards and epsilon. 
        rewards_per_episode = []

        # Track number of steps taken. Used for syncing policy => target network.
        best_rewards = - float('inf')
            
        for episode in range(1, NUM_EPISODES):
            state = self.env.reset(episode)
            terminated = False   # True when agent falls in hole or reached goal
            rewards = 0
            step = 0

            # Agent navigates map until it collides a wall/reaches goal (terminated), or has taken 200 actions (truncated).
            while not terminated and step < 200:

                # Select action based on epsilon-greedy
                if rd.random() < epsilon:
                    action = [
                    rd.choices(self.env.action_space, weights=[0.5, 0.2, 0.2, 0.1])[0] if car.alive else None for car in self.env.car_list
                    ] # on favorise l'action up
                    #print('random', action)
                else:
                    with torch.no_grad():
                        action = [
                            policy_dqn(self.normalisation(state[i])).argmax().item() if car.alive else None
                            for i, car in enumerate(self.env.car_list)
                        ]

                if len(action) != len(self.env.car_list):
                    raise ValueError(
                        f"Error: actions ({len(action)}) and car_list ({len(self.env.car_list)}) lengths do not match.")

                        # print('nn',action)
                #if not isinstance(action, (list, tuple)):
                #   action = [action] * len(self.env.action_space)


                # Execute action
                new_state, reward, terminated = self.env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))
                #print((state, action, new_state, reward, terminated))

                state = new_state
                rewards += sum(reward)
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
            if len(memory) > BATCH_SIZE:
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

        for state, action, new_state, reward, terminated in mini_batch:


            if terminated: 
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward]*len(action))
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + GAMMA * target_dqn(self.normalisation(new_state)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.normalisation(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.normalisation(state)).detach()
            # Adjust the specific action to the target that was just calculated
            for i in range(len(action)):
                if action[i] is not None: #on ignore si la voiture est morte
                    if target.dim() > 0:
                        target_q[i, action[i]] = target[i].squeeze() #réduit les dimensions
                    else:
                        target_q[i, action[i]] = target
            #target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def normalisation(self, state):
        """ Scales nn inputs to [-1, 1] for better convergence """
        # Vérifier si `state` est un entier ou une valeur incorrecte
        if isinstance(state, (int, float)):
            # Si state est un entier/flottant, supposons qu'il représente une seule valeur d'état
            state = [state]  # Convertir en liste

        # Vérifier que `state` est une liste ou un tuple
        if not isinstance(state, (list, tuple)):
            raise TypeError(f"`state` doit être une liste ou un tuple, mais un {type(state)} a été fourni : {state}")

        # Vérifier que observation_space est bien formaté
        if not isinstance(self.env.observation_space, (list, tuple)):
            raise TypeError(
                f"`self.env.observation_space` doit être une liste ou un tuple, mais un {type(self.env.observation_space)} a été fourni."
            )

        # Normaliser chaque composante de l'état
        normalized_state = []
        for i in range(len(state)):
            # Vérifier que observation_space[i] est un tuple attendu
            if not isinstance(self.env.observation_space[i], (list, tuple)) or len(self.env.observation_space[i]) != 2:
                raise ValueError(
                    f"self.env.observation_space[{i}] doit être un tuple (min, max), mais a été {self.env.observation_space[i]}"
                )

            # Appliquer la fonction de mise à l'échelle
            normalized_value = lib.scale(state[i], *self.env.observation_space[i])
            normalized_state.append(normalized_value)

        return torch.FloatTensor(normalized_state)

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



if __name__ == '__main__':
    filepath = "results_dql/weights.pt"

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