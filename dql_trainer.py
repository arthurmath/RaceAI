import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from dql_game import Session
import library as lib



LR = 0.01                    # learning rate
GAMMA = 0.95                 # discount rate
SYNC_RATE = 500              # number of steps the agent takes before syncing target with policy network
BATCH_SIZE = 32              # size of the data set sampled from the replay memory
EPS_DECAY = 4                # epsilon decay rate 
EPS_MIN = 0.1                # minimum value of epsilon
NUM_EPISODES = 10_000        # number of episodes
MEMORY_LEN = 10_000          # size of the buffer
PLOT_RATE = 100              # when to plot rewards

rd.seed(0)




class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 20)   # fully-connected layer
        self.fc2 = nn.Linear(20, 20)       # fully-connected layer
        self.out = nn.Linear(20, outputs)  # ouptut layer

    def forward(self, x):
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
        self.env = Session(nb_cars=1, display=render)
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
                    action = rd.choices(self.env.action_space, weights=[0.5, 0.2, 0.2, 0.1])[0] # on favorise l'action up
                    #print('random', action)
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.normalisation(state)).argmax().item()
                        # print('nn',action)

                # Execute action
                new_state, reward, terminated = self.env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))
                #print((state, action, new_state, reward, terminated))

                state = new_state
                rewards += reward
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
                target = torch.FloatTensor([reward])
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
            target_q[action] = target
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
        state = [lib.scale(state[i], *self.env.observation_space[i]) for i in range(len(state))]
        return torch.FloatTensor(state)
    
            
    def plot_progress(self, rewards):
        win = 100
        x = np.arange(int(win / 2), len(rewards)-int(win / 2))
        plt.figure()
        plt.plot(rewards)
        plt.plot(x, lib.window_average(rewards, win), color='black')
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