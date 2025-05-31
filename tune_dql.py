from sklearn.model_selection import RandomizedSearchCV
from dql_sklearn import DQLWrapper
import numpy as np

# Plages de valeurs à tester
param_dist = {
    'lr':                [1e-2, 1e-3],
    'gamma':             [0.90, 0.95, 0.99],
    'batch_size':        [32, 64, 128],
    'epsilon_decay_rate':[0.99, 0.995, 0.999,0.997],
    'num_episodes':      [400, 700, 1000],
}
def main():
    search = RandomizedSearchCV(
        estimator=DQLWrapper(),
        param_distributions=param_dist,
        n_iter=10,            # nombre de configurations tirées au hasard
        scoring='neg_mean_squared_error',
        cv=None,                 # pas de cross‑val pour le RL
        verbose=2,
        n_jobs=1              # RL est coûteux, on évite trop de parallélisme
)
    X = np.zeros((10, 5))
    y = np.zeros((10,))
# on n'a pas de X, y : on passe None
    search.fit(X, y)



    print("Meilleurs hyper‑paramètres :", search.best_params_)
    print("Meilleur score (20 derniers rewards) :", search.best_score_)
    print("Meilleur score :", search.best_score_)
    print("Meilleur modèle :", search.best_estimator_)
    print("Meilleur modèle (agent) :", search.best_estimator_.agent)
    print("Meilleur modèle (rewards) :", search.best_estimator_.rewards_per_episode)
if __name__ == "__main__":
    main()


##Resultats premier lancement :
#
# Meilleur score (20 derniers rewards) : nan
# Meilleur score : nan
# Meilleur modèle : DQLWrapper(batch_size=32, epsilon_decay_rate=0.99, gamma=0.99, num_episodes=100)
# Meilleur modèle (agent) : <sandbox_1.DQL object at 0x000001950B64B550>
# Meilleur modèle (rewards) : [-1294518.6688542268, -1361164.2085528406, -1136161.2978713273, -1358087.6164722748, -1339385.073165342, -1158778.2595313306, -1426813.045744771, -1062661.4910380614, -1133915.5652877237]


#  Meilleurs hyper‑paramètres : {'num_episodes': 100, 'lr': 0.001, 'gamma': 0.99, 'epsilon_decay_rate': 0.99, 'batch_size': 64}
# Meilleurs hyper‑paramètres : {'num_episodes': 400, 'lr': 0.001, 'gamma': 0.9, 'epsilon_decay_rate': 0.997, 'batch_size': 128}
# Meilleur score (20 derniers rewards) : nan
# Meilleur score : nan
# Meilleur modèle : DQLWrapper(batch_size=128, epsilon_decay_rate=0.997, gamma=0.9,
#            num_episodes=400)