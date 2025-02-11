import numpy as np
import random
import gym
from kaggle_environments import make, evaluate


# https://www.kaggle.com/code/alexisbcook/deep-reinforcement-learning


def agent_random(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)



# There are a lot of great implementations of reinforcement learning algorithms online. In this course, we'll use Stable-Baselines3.
# We have to make the environment compatible with Stable Baselines. For this, we define the ConnectFourGym class below. 
# This class implements ConnectFour as an OpenAI Gym environment.
# Learn about spaces here: http://gym.openai.com/docs/#spaces



class ConnectFourGym(gym.Env):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(1,self.rows,self.columns), dtype=int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
        
    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns)
    
    def change_reward(self, old_reward, done):
        if old_reward == 1: # The agent won the game
            return 1
        elif done: # The opponent won the game
            return -1
        else: # Reward 1/42 (better convergence)
            return 1/(self.rows*self.columns)
        
    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid: # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else: # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns), reward, done, _
    
    
    
# Create ConnectFour environment 
env = ConnectFourGym(agent2="random")






# The next step is to specify the architecture of the neural network. In this case, we use a convolutional neural network. 
# There are many different reinforcement learning algorithms, such as DQN, A2C, and PPO, among others. 
# All of these algorithms use a similar process to produce an agent:
# - Initially, the weights are set to random values.
# - As the agent plays the game, the algorithm continually tries out new values for the weights, to see how the cumulative reward is affected. 
# - Over time, after playing many games, the algorithm settles towards weights that got high rewards (performs better).
# To learn more about how to specify architectures with Stable-Baselines3 : https://stable-baselines3.readthedocs.io/en/master/



import torch as th
import torch.nn as nn
from stable_baselines3 import PPO # !pip install "stable-baselines3"
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



# Neural network for predicting action values
class CustomCNN(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(features_extractor_class=CustomCNN,)




# Initialize agent
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

# Train agent
model.learn(total_timesteps=50000)







# Learn more about the stable-baselines3 package to amend the agent.
# Change agent2 to a different agent when creating the ConnectFour environment 
# with env = ConnectFourGym(agent2="random"). For instance, you might like to use the "negamax" agent, 
# or a different, custom agent. Note that the smarter you make the opponent, the harder it will be for your agent to train!