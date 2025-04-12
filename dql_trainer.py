import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from dql_game import Session


class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 10)   # fully-connected layer
        self.fc2 = nn.Linear(10, 10)       # fully-connected layer
        self.out = nn.Linear(10, outputs)  # ouptut layer

    def forward(self, x):
        x = F.relu(self.fc1(x))     # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))     # Calculate output
        return x


class ReplayMemory():
    def __init__(self):
        self.memory = deque([], maxlen=100_000)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class DQL():
    # Hyperparameters 
    alpha = 0.01                    # learning rate
    gamma = 0.9                     # discount rate  
    network_sync_rate = 500         # number of steps the agent takes before syncing the policy and target network
    batch_size = 32                 # size of the data set sampled from the replay memory
    num_divisions = 30
    min_eps = 0
    nb_episodes = 100
    
    def __init__(self, render):
        self.env = Session(nb_cars=1, display=render)
        self.num_states = len(self.env.observation_space) # 4: x, y, speed, angle
        self.num_actions = len(self.env.action_space) # 4: up, down, left, right

        # Divide position and velocity into segments
        self.x_space = np.linspace(self.env.observation_space[0][0], self.env.observation_space[0][1], self.num_divisions) 
        self.y_space = np.linspace(self.env.observation_space[1][0], self.env.observation_space[1][1], self.num_divisions)
        self.speed_space = np.linspace(self.env.observation_space[2][0], self.env.observation_space[2][1], self.num_divisions)
        self.angle_space = np.linspace(self.env.observation_space[3][0], self.env.observation_space[3][1], self.num_divisions)


    def train(self, filepath):
        epsilon = 1 # 100% random actions
        memory = ReplayMemory()

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(self.num_states, self.num_actions)
        target_dqn = DQN(self.num_states, self.num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict()) # Copy weights/biases from policy to the target
        
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        # List to keep track of rewards and epsilon. 
        rewards_per_episode = []
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        best_rewards = - float('inf')
        goal_reached = False
            
        for episode in range(self.nb_episodes):
            state = self.env.reset()
            terminated = False   # True when agent falls in hole or reached goal
            rewards = 0
            step_count = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while not terminated:

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    action = np.random.choice(self.env.action_space, p=[4/6, 1/6, 1/6, 0]) # on favorise l'action up
                else:       
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                new_state, reward, terminated = self.env.step(action, step_count)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state
                
                rewards += reward
                step_count += 1
                
                if self.env.episode_done:
                    break

            if self.env.quit:
                break
            
            rewards_per_episode.append(rewards) 
            
            if terminated:
                goal_reached = True

            # Graph training progress
            if episode != 0 and episode % 200 == 0 :
                print(f'Episode {episode}, epsilon {epsilon}, reward: {rewards}')
                self.plot_progress(rewards_per_episode, epsilon_history)
            
            if rewards > best_rewards:
                best_rewards = rewards
                torch.save(policy_dqn.state_dict(), filepath)

            # Check if enough experience has been collected
            if len(memory) > self.batch_size and goal_reached:
                mini_batch = memory.sample(self.batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 2 / self.nb_episodes, self.min_eps)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of episodes
                if episode % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

        self.env.close()
        

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.gamma * target_dqn(self.state_to_dqn_input(new_state)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def state_to_dqn_input(self, state) -> torch.Tensor:
        ''' Converts a state (position, velocity) to tensor representation.
        Example: Input = (0.3, -0.03) -> Return = tensor([16, 6]) '''
        state_x = np.digitize(state[0], self.x_space)
        state_y = np.digitize(state[1], self.y_space)
        state_s = np.digitize(state[2], self.speed_space)
        state_a = np.digitize(state[3], self.angle_space)
        return torch.FloatTensor([state_x, state_y, state_s, state_a])
    
            
    def plot_progress(self, rewards_per_episode, epsilon_history):
        plt.figure(1)
        plt.subplot(121) # plot on a 1 row * 2 col grid, at cell 1
        plt.plot(rewards_per_episode)
        plt.subplot(122) # cell 2
        plt.plot(epsilon_history)
        plt.savefig(f'raceAI_dql_{len(epsilon_history)}.png')
    
    
    # Run the environment with the learned policy
    def test(self, filepath):

        # Load learned policy
        policy_dqn = DQN(self.num_states, self.num_actions) 
        policy_dqn.load_state_dict(torch.load(filepath))
        policy_dqn.eval()    # switch model to evaluation mode

        state = self.env.reset()
        terminated = False      # True when agent falls in hole or reached goal
        step_count = 0          

        while not terminated: 
            with torch.no_grad():
                action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

            state, reward, terminated = self.env.step(action, step_count)
            step_count += 1
                
        self.env.close()



if __name__ == '__main__':
    filepath = "weights.pt"

    mountaincar = DQL(render=True)
    # mountaincar.train(True, filepath)
    mountaincar.test(filepath)
    
    
    
    


# if __name__ == '__main__':
    
#     dqn = DeepQNetwork()
#     with open(Path("results_dqn/weights/colab1.weights"), "rb") as f:
#         dqn.model.set_weights(pickle.load(f))
        
#     env = Session(nb_cars=1)
#     states = env.reset()
#     while not env.quit:
#         actions = dqn.model.predict(np.array(states)[np.newaxis, :], verbose=0)[0].argmax()
#         states = env.step(actions)
        
