import matplotlib.pyplot as plt
from gene_pilot import Pilot
from gene_game import Session
from pathlib import Path
import copy as cp
import random as rd
import pickle
import os


SEED = 42
POPULATION = 500
SURVIVAL_RATE = 0.1
N_EPISODES = 100
N_STEPS = 120 
STEPS_INCREASE = 4

STD_MUTATION = 0.2
MUTATION_RATE = 0.1
MR_MIN = 0.1
MR_FACTOR = int(N_EPISODES * 1) 

rd.seed(SEED)





class GeneticAlgo:

    def train(self):
        
        self.best_scores = []
        self.avg_scores = []
        self.mutation_rate = 1
        self.threshold = 400
        
        self.ses = Session(self, display=True, nb_cars=POPULATION)
        
        self.population = [Pilot() for _ in range(POPULATION)]

        for self.generation in range(N_EPISODES):
            
            self.evaluate_generation()        
            self.bests_survives()
            self.change_generation()
            
            if self.ses.quit:
                break
            
            print(f"Generation {self.generation+1}, avg score: {self.avgGenScore:.2f}, best score: {self.bestGenScore:.2f}") # , mr: {self.mutation_rate:.2f}
            
        if not self.ses.quit:
            self.evaluate_generation() # Evaluate the last generation
            self.bests_survives()
            self.bestPilotEver = self.bestPilots[0]
            
            # print(f"BEST SCORE : {self.best_scores[0]:.3f}")
        


    def evaluate_generation(self):
            
        self.ses.reset(self.generation, self.population[0])
        states = self.ses.get_states()

        for step in range(N_STEPS + STEPS_INCREASE * self.generation):
            
            actions = [self.population[i].predict(states[i]) for i in range(len(self.population))]
        
            states = self.ses.step(actions)
            
            # print("STATES :", states)
            # print("ACTIONS : ", actions)
            
            if self.ses.done:
                break
            
        self.scores = self.ses.get_scores()
        self.bestGenScore = max(self.scores)
        self.avgGenScore = sum(self.scores) / POPULATION
        self.best_scores.append(self.bestGenScore)
        self.avg_scores.append(self.avgGenScore)
        

            
    def bests_survives(self):
        
        sorted_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)
        
        population_sorted = [self.population[i] for i in sorted_indices] 
        scores_sorted = [self.scores[i] for i in sorted_indices] 
        
        self.survival_prop = int(POPULATION * SURVIVAL_RATE) # 50
        
        self.bestPilots = population_sorted[:self.survival_prop] # take the 10% bests pilots
        self.best_scores = scores_sorted[:self.survival_prop]  # take the 10% bests scores
        


                
    def change_generation(self):
        """ Creates a new generation of pilot. """
        
        self.new_population = cp.deepcopy(self.bestPilots) # 10% best pilots
        
        while len(self.new_population) < POPULATION:
            self.mutation_rate = max(1 - self.generation / MR_FACTOR, MR_MIN)
            
            if len(self.new_population) < self.threshold:
                parent1, parent2 = cp.deepcopy(self.select_parents_bests()) # blue
                baby = parent1.mate(parent2)
                baby.mutate(0.3, std=0.1)
            else:
                baby = cp.deepcopy(rd.choices(self.bestPilots[:5])[0]) # green
                baby.mutate(0.1, std=0.1)
                
            self.new_population.append(baby)
        
        self.population = self.new_population

        
    
    def select_parents_bests(self):
        """Select two pilots with high scores among best ones."""
        total_scores = sum(self.best_scores)
        ratios = [f / total_scores for f in self.best_scores]
        return rd.choices(self.bestPilots, weights=ratios, k=2) # return a k-sized list 
    










if __name__ == "__main__":
    
    
    algo = GeneticAlgo()
    algo.train()
    
    
    
    if not algo.ses.quit:
        # Save weights and biases of the best pilot
        PATH = Path("results_gene/weights")
        n_train = len(os.listdir(PATH)) # nb de fichiers dans dossier weights
        with open(PATH / Path(f"{n_train}.weights"), "wb") as f: # write binary
            pickle.dump((algo.bestPilotEver.weights, algo.bestPilotEver.bias), f)
    
        # # Show graph of scores
        # plt.plot(algo.best_scores, label='Best scores')
        # plt.plot(algo.avg_scores, label='Average scores')
        # plt.xlabel("Générations")
        # plt.ylabel("Scores (%)")
        # plt.legend()
        # plt.show()
        
             
            

        




# Base: threshold = 400, mutate(0.3, std=0.1), mutate(0.1, std=0.1), bestPilots[:5], [5, 10, 10, 4]
# Generation 1, avg score: 0.56, best score: 5.42
# Generation 2, avg score: 2.22, best score: 11.07
# Generation 3, avg score: 3.03, best score: 12.31
# Generation 4, avg score: 3.12, best score: 13.69
# Generation 5, avg score: 4.23, best score: 14.17
# Generation 6, avg score: 4.91, best score: 15.72
# ------------------------------------------------
# Generation 13, avg score: 7.65, best score: 15.72
# Generation 14, avg score: 7.75, best score: 17.78
# Generation 15, avg score: 7.77, best score: 17.78
# Generation 16, avg score: 7.73, best score: 17.78
# Generation 17, avg score: 8.03, best score: 21.72
# Generation 18, avg score: 7.51, best score: 21.72
# Generation 19, avg score: 7.88, best score: 22.67
# Generation 20, avg score: 7.89, best score: 25.75
# Generation 21, avg score: 8.15, best score: 25.79
# Generation 22, avg score: 8.12, best score: 25.79
# Generation 23, avg score: 8.72, best score: 25.79
# Generation 24, avg score: 8.42, best score: 25.79
# Generation 25, avg score: 8.70, best score: 27.55
# Generation 26, avg score: 8.31, best score: 27.55
# Generation 27, avg score: 8.22, best score: 28.69
# Generation 28, avg score: 8.39, best score: 31.97
# Generation 29, avg score: 8.27, best score: 31.97
# Generation 30, avg score: 8.24, best score: 31.97
# Generation 31, avg score: 8.65, best score: 31.97
# Generation 32, avg score: 8.62, best score: 32.14
# Generation 33, avg score: 8.26, best score: 32.14


# no mate: mutate(0.2, std=0.1)
# Generation 1, avg score: 0.56, best score: 5.42
# Generation 2, avg score: 2.11, best score: 11.07
# Generation 3, avg score: 2.99, best score: 12.31
# Generation 4, avg score: 3.02, best score: 13.69
# Generation 5, avg score: 3.50, best score: 13.69
# Generation 6, avg score: 3.91, best score: 17.78
# Generation 7, avg score: 4.36, best score: 17.78
# Generation 8, avg score: 4.34, best score: 24.12
# Generation 9, avg score: 4.64, best score: 25.79
# ------------------------------------------------
# Generation 31, avg score: 7.79, best score: 25.79


# no mate: mutate(0.1, std=0.2)
# Generation 1, avg score: 0.56, best score: 5.42
# Generation 2, avg score: 2.06, best score: 11.07
# Generation 3, avg score: 2.82, best score: 12.31
# Generation 4, avg score: 2.83, best score: 14.54
# Generation 5, avg score: 3.42, best score: 15.72
# Generation 6, avg score: 4.33, best score: 15.72
# Generation 7, avg score: 4.68, best score: 20.10
# Generation 8, avg score: 4.81, best score: 20.57
# Generation 9, avg score: 4.56, best score: 20.57
# Generation 10, avg score: 5.08, best score: 20.57
# Generation 11, avg score: 5.14, best score: 20.70
# Generation 12, avg score: 5.31, best score: 24.12
# Generation 13, avg score: 5.70, best score: 24.12
# Generation 14, avg score: 5.78, best score: 24.12
# Generation 15, avg score: 6.01, best score: 24.12
# Generation 16, avg score: 6.33, best score: 24.12
# Generation 17, avg score: 6.52, best score: 25.59
# Generation 18, avg score: 6.40, best score: 25.59
# Generation 19, avg score: 6.40, best score: 25.68
# Generation 20, avg score: 6.62, best score: 25.68
# Generation 21, avg score: 6.61, best score: 25.79
# -------------------------------------------------
# Generation 32, avg score: 7.85, best score: 25.79


# threshold : 200
# Generation 1, avg score: 0.56, best score: 5.42
# Generation 2, avg score: 3.61, best score: 5.42
# Generation 3, avg score: 3.95, best score: 5.42
# Generation 4, avg score: 4.05, best score: 5.42
# Generation 5, avg score: 3.96, best score: 7.38
# Generation 6, avg score: 4.10, best score: 7.38
# Generation 7, avg score: 5.61, best score: 10.69
# Generation 8, avg score: 6.60, best score: 11.15
# Generation 9, avg score: 6.92, best score: 11.46
# Generation 10, avg score: 7.41, best score: 12.41
# Generation 11, avg score: 7.63, best score: 15.72
# -------------------------------------------------
# Generation 20, avg score: 8.15, best score: 15.72


# threshold : 300
# Generation 1, avg score: 0.56, best score: 5.42
# Generation 2, avg score: 2.90, best score: 11.07
# Generation 3, avg score: 3.66, best score: 14.82
# Generation 4, avg score: 4.57, best score: 17.78
# Generation 5, avg score: 6.08, best score: 17.78
# Generation 6, avg score: 5.88, best score: 17.78
# Generation 7, avg score: 6.44, best score: 17.79
# Generation 8, avg score: 6.92, best score: 18.93
# Generation 9, avg score: 6.19, best score: 18.93
# Generation 10, avg score: 6.28, best score: 21.84
# Generation 11, avg score: 6.16, best score: 21.84
# Generation 12, avg score: 6.36, best score: 25.79
# -------------------------------------------------
# Generation 44, avg score: 7.22, best score: 25.79


# rd.choices(self.bestPilots[:10])
# Generation 1, avg score: 0.56, best score: 5.42
# Generation 2, avg score: 2.06, best score: 11.07
# Generation 3, avg score: 2.76, best score: 12.31
# Generation 4, avg score: 2.86, best score: 13.69
# Generation 5, avg score: 3.64, best score: 15.72
# Generation 6, avg score: 4.48, best score: 15.72
# Generation 7, avg score: 5.04, best score: 20.37
# Generation 8, avg score: 5.63, best score: 20.37
# Generation 9, avg score: 6.86, best score: 20.72
# Generation 10, avg score: 7.24, best score: 21.60
# Generation 11, avg score: 7.50, best score: 25.79
# -------------------------------------------------
# Generation 33, avg score: 8.87, best score: 25.79


# NN_LAYERS = [5, 6, 6, 4]
# Generation 1, avg score: 0.56, best score: 5.42
# Generation 2, avg score: 2.47, best score: 9.19
# -------------------------------------------------
# Generation 16, avg score: 4.70, best score: 9.19






# Entrainement avec select_best only : best score meilleur : 25.5%, avg : 9.68
# Avoir un mutation rate petit augmente l'average score mais diminue le best score final
# Etre plus elitiste : prendre le meilleur et lui appliquer de toute petites mutations
# BestPilots only et juste mutate : mauvais résultats (gen:50, avg:3, best:7.2) std_mutation inversé !! à refaire

# Améliorations : 
# Tuer les cars qui ont un score < 5% pour accélérer le temps de train global 
# Ne pas réévaluer les 50 meilleurs pilotes (inutile) mais les conserver dans une 2e liste. 




# print(all([all([(self.bestPilots[j].weights[i] == self.new_population[j].weights[i]).all() for i in range(3)]) for j in range(50)]))
# Solution au problème de baisse du high score entre 2 generations : il faut deepcopy quand on sélectionne les parents, sinon la mutation les modifie aussi 