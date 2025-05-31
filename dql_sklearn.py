import numpy as np
from sklearn.base import BaseEstimator
#from dql_trainer_multi_1_tensorflow import DQL  # votre agent DQL
from sandbox_1 import DQL
from dql_game_multi_1 import Session
from sandbox_1 import ReplayMemory
from sandbox_1 import DQN

class DQLWrapper(BaseEstimator):
    def __init__(self,
                 lr=0.001,
                 gamma=0.95,
                 batch_size=64,
                 epsilon_decay_rate=0.995,
                 num_episodes=50,
                 population_size=70):
        # on stocke tous les hyper‑paramètres
        self.lr                  = lr
        self.gamma               = gamma
        self.batch_size          = batch_size
        self.epsilon_decay_rate  = epsilon_decay_rate
        self.num_episodes        = num_episodes
        self.population_size     = population_size

        # on passe ces hyper‑paramètres à l'agent
        self.agent = DQL(render=False,
                         lr=lr,
                         gamma=gamma,
                         batch_size=batch_size,
                         epsilon_decay_rate=epsilon_decay_rate,
                         num_episodes=num_episodes,
                         population_size=population_size)

        # on prépare l’attribut scikit‑learn
        self.rewards_per_episode = None

    def fit(self, X=None, y=None):
        # 1) Lancer l’entraînement
        self.agent.train("tmp_weights_1")
        # 2) Récupérer la liste des rewards pour le score
        self.rewards_per_episode = self.agent.rewards_per_episode
        return self

    def score(self, X=None, y=None):
        # par exemple, moyenne des 20 derniers rewards
        return float(np.mean(self.rewards_per_episode[-20:]))

if __name__ == "__main__":
    wrapper = DQLWrapper()
    wrapper.fit()
    print("Rewards par épisode :", wrapper.rewards_per_episode)
    print("Score moyen (20 derniers) :", wrapper.score())
