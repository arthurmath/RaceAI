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
N_EPISODES = 20
N_STEPS = 100 
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
        
        self.ses = Session(display=True, nb_cars=POPULATION)
        
        self.population = [Pilot() for _ in range(POPULATION)]

        for self.generation in range(N_EPISODES):
            
            self.evaluate_generation()        
            self.bests_survives()
            self.change_generation()
            
            if self.ses.quit:
                break
            
            print(f"Generation {self.generation+1}, avg score: {self.avgGenScore:.2f}, best score: {self.bestGenScore:.2f}") # , mr: {self.mutation_rate:.2f}
            print()
            
        if not self.ses.quit:
            self.evaluate_generation() # Evaluate the last generation
            self.bests_survives()
            self.bestPilotEver = self.bestPilots[0]
            
            # print(f"BEST SCORE : {self.best_scores[0]:.3f}")
        


    def evaluate_generation(self):
        self.scores = []
            
        self.ses.reset(self.generation)
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
            
            if len(self.new_population) < 400:
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
        
             
            

        







# Base : mut_rate=0.9, cross_layer, std_mut = 0.5

# mutation_rate = 0.1
# Generation 1, average score: 0.65, best score: 5.66
# Generation 2, average score: 1.92, best score: 7.20
# Generation 3, average score: 2.33, best score: 7.16
# Generation 4, average score: 2.41, best score: 8.98
# Generation 5, average score: 2.61, best score: 9.17
# Generation 6, average score: 2.66, best score: 9.03
# Generation 7, average score: 2.82, best score: 8.87
# Generation 8, average score: 2.91, best score: 8.80
# Generation 9, average score: 2.79, best score: 8.76
# Generation 10, average score: 3.00, best score: 9.09

# mutation_rate = 0.5
# Generation 1, average score: 0.65, best score: 5.66
# Generation 2, average score: 1.78, best score: 7.98
# Generation 3, average score: 2.11, best score: 9.34
# Generation 4, average score: 2.20, best score: 9.45
# Generation 5, average score: 2.28, best score: 9.53
# Generation 6, average score: 2.41, best score: 8.00
# Generation 7, average score: 2.44, best score: 8.74
# Generation 8, average score: 2.45, best score: 8.57
# Generation 9, average score: 2.33, best score: 8.39
# Generation 10, average score: 2.35, best score: 8.37

# mutation_rate = 0.9
# Generation 1, average score: 0.65, best score: 5.66
# Generation 2, average score: 1.65, best score: 7.16
# Generation 3, average score: 1.92, best score: 7.62
# Generation 4, average score: 2.17, best score: 7.69
# Generation 5, average score: 2.40, best score: 9.17
# Generation 6, average score: 2.21, best score: 13.74
# Generation 7, average score: 2.43, best score: 13.77
# Generation 8, average score: 2.43, best score: 13.83
# Generation 9, average score: 2.16, best score: 13.88
# Generation 10, average score: 2.40, best score: 13.85

# cross_layer2
# Generation 1, average score: 0.65, best score: 5.66
# Generation 2, average score: 1.28, best score: 5.26
# Generation 3, average score: 1.48, best score: 5.32
# Generation 4, average score: 1.85, best score: 8.77
# Generation 5, average score: 2.04, best score: 7.82
# Generation 6, average score: 2.11, best score: 7.55
# Generation 7, average score: 2.30, best score: 10.47
# Generation 8, average score: 2.18, best score: 8.45
# Generation 9, average score: 2.08, best score: 7.16
# Generation 10, average score: 2.36, best score: 11.40

# std_mutation = 0.1
# Generation 1, average score: 0.65, best score: 5.66
# Generation 2, average score: 1.92, best score: 7.83
# Generation 3, average score: 2.17, best score: 8.16
# Generation 4, average score: 2.47, best score: 8.04
# Generation 5, average score: 2.55, best score: 7.92
# Generation 6, average score: 2.59, best score: 8.36
# Generation 7, average score: 2.62, best score: 8.44
# Generation 8, average score: 2.68, best score: 8.52
# Generation 9, average score: 2.74, best score: 8.57
# Generation 10, average score: 2.77, best score: 10.46

# std_mutation = 0.2
# Generation 1, average score: 0.65, best score: 5.66
# Generation 2, average score: 1.86, best score: 5.54
# Generation 3, average score: 2.17, best score: 7.59
# Generation 4, average score: 2.35, best score: 13.80
# Generation 5, average score: 2.42, best score: 13.81
# Generation 6, average score: 2.65, best score: 15.83
# Generation 7, average score: 2.65, best score: 15.81
# Generation 8, average score: 2.74, best score: 15.79
# Generation 9, average score: 2.61, best score: 15.77
# Generation 10, average score: 2.59, best score: 15.76

# std_mutation = 0.3
# Generation 1, average score: 0.65, best score: 5.66
# Generation 2, average score: 1.90, best score: 7.89
# Generation 3, average score: 2.17, best score: 9.18
# Generation 4, average score: 2.27, best score: 9.11
# Generation 5, average score: 2.33, best score: 9.02
# Generation 6, average score: 2.44, best score: 9.03
# Generation 7, average score: 2.56, best score: 9.48
# Generation 8, average score: 2.50, best score: 9.46
# Generation 9, average score: 2.47, best score: 9.44
# Generation 10, average score: 2.71, best score: 9.40



# Entrainement avec select_best only : best score meilleur : 25.5%, avg : 9.68
# Avoir un mutation rate petit augmente l'average score mais diminue le best score final
# Etre plus elitiste : prendre le meilleur et lui appliquer de toute petites mutations
# BestPilots only et juste mutate : mauvais résultats (gen:50, avg:3, best:7.2) std_mutation inversé !! à refaire

# Améliorations : 
# Tuer les cars qui ont un score < 5% pour accélérer le temps de train global 
# Ne pas réévaluer les 50 meilleurs pilotes (inutile) mais les conserver dans une 2e liste. 




# print(all([all([(self.bestPilots[j].weights[i] == self.new_population[j].weights[i]).all() for i in range(3)]) for j in range(50)]))
# Solution au problème de baisse du high score entre 2 generations : il faut deepcopy quand on sélectionne les parents, sinon la mutation les modifie aussi 