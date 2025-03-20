import random as rd
import pickle
import os
import matplotlib.pyplot as plt
from gene_pilot import Pilot
from gene_game import Session
from pathlib import Path
import copy as cp


SEED = 42
POPULATION = 500
SURVIVAL_RATE = 0.1
N_EPISODES = 20
N_STEPS = 100    
EPISODE_INCREASE = 2

rd.seed(SEED)





class GeneticAlgo:

    def train(self):
        
        self.list_scores = []
        
        self.population = [Pilot() for _ in range(POPULATION)]

        for self.generation in range(N_EPISODES):
            
            self.evaluate_generation()        
            self.bests_survives()
            self.change_generation()
            
            if self.ses.done:
                break
            
            print(f"Generation {self.generation+1}, average score: {self.avgGenScore:.2f}, best score: {self.bestGenScore:.2f}\n")
            
        # self.evaluate_generation() # Evaluate the last generation
        # self.bests_survives()
        # self.bestPilotEver = self.bestPilots[-1]
        


    def evaluate_generation(self):
        self.scores = []
            
        self.ses = Session(display=True, nb_cars=POPULATION, gen=self.generation)
        states = self.ses.get_states()

        for step in range(N_STEPS + EPISODE_INCREASE * self.generation):
            
            actions = [self.population[i].predict(states[i]) for i in range(len(self.population))]
            actions = [mat.tolist()[0] for mat in actions]
            actions = [[j for j, act in enumerate(action) if act] for action in actions]
        
            states = self.ses.step(actions)
            
            # print("STATES :", states)
            # print("ACTIONS : ", actions)
            
            if self.ses.done:
                break
            
        self.scores = self.ses.get_scores()
            
        self.bestGenScore = max(self.scores)
        self.avgGenScore = sum(self.scores) / POPULATION
        self.list_scores.append(self.bestGenScore)
        

            
    def bests_survives(self):
        
        sorted_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)
        
        population_sorted = [self.population[i] for i in sorted_indices] 
        scores_sorted = [self.scores[i] for i in sorted_indices] 
        
        self.bestPilots = population_sorted[:int(POPULATION * SURVIVAL_RATE)] # take the 10% bests pilots
        self.bestscores = scores_sorted[:int(POPULATION * SURVIVAL_RATE)]  # take the 10% bests scores
        
        print([round(x, 2) for x in self.bestscores])
                

                
    def change_generation(self):
        """ Creates a new generation of pilot. """
        
        self.new_population = cp.copy(self.bestPilots) # 10% best pilots
        
        while len(self.new_population) < POPULATION:
            parent1, parent2 = self.select_parents()
            baby = parent1.mate(parent2)
            baby.mutate()
            self.new_population.append(baby)
        
        self.population = self.new_population
        
    
    def select_parents(self):
        """Select two pilots with high scores."""
        total_scores = sum(self.bestscores)
        ratios = [f / total_scores for f in self.bestscores]
        return rd.choices(self.bestPilots, weights=ratios, k=2) # return a k-sized list










if __name__ == "__main__":
    
    
    algo = GeneticAlgo()
    algo.train()
    
    print(f"\nBests scores total: {sum(algo.list_scores):.2f}\n")
    
    
    
    # # Save the weights and biases of the snakes for the new game scores
    # files = os.listdir(Path("weights"))
    # n_train = len(files) # nb de fichiers dans dossier weights
    # with open(Path("weights") / Path(f"{n_train}.weights"), "wb") as f: # write binary
    #     pickle.dump((algo.bestPilotEver.weights, algo.bestPilotEver.bias), f)
        
        
    
    # # Show graph of progressions
    # plt.plot(algo.list_scores)
    # plt.xlabel("Générations")
    # plt.ylabel("Progression (%)")
    # plt.show()
        
            
            




# Pourquoi les meilleurs pilotes ne performent pas aussi bien à la génération suivante ? 

# Mes reward favorisent les pilotes à avancer peu car ceux qui vont vitent meurent vite