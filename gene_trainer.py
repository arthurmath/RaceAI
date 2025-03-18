import random as rd
import pickle
import os
import matplotlib.pyplot as plt
from gene_pilot import Pilot
from gene_game import Session
from pathlib import Path
import copy as cp
import pygame as pg
pg.init()



N_EPISODES = 10
N_STEPS = 200    
POPULATION = 500
SURVIVAL_RATE = 0.1
    
# Autres paramètres :
# nombre de layers NN
# fonctions activation 



class GeneticAlgo:

    def train(self):
        
        self.list_scores = []
        
        self.population = [Pilot() for _ in range(POPULATION)]

        for generation in range(N_EPISODES):
            
            self.evaluate_generation()        
            self.bests_survives()
            self.change_generation()
            
            print(f"Generation {generation+1}, average score: {self.avgGenScore:.0f}, best score: {self.bestGenScore:.2f}")
            
        self.evaluate_generation() # Evaluate the last generation
        self.bests_survives()
        self.bestPilotEver = self.bestPilots[-1]
        


    def evaluate_generation(self):
        self.scores = []
            
        ses = Session(train=True, display=True, training_time=None, nb_cars=POPULATION)
        states = ses.reset()

        for step in range(N_STEPS):
            
            actions = [self.population[i].predict(states[i]) for i in range(len(self.population))]
            actions = [mat.tolist()[0] for mat in actions]
            states, self.scores = ses.step(actions)
            
        self.bestGenScore = max(self.scores)
        self.avgGenScore = sum(self.scores) / POPULATION
        self.list_scores.append(self.bestGenScore)
        


            
    def bests_survives(self):
        
        sorted_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)
        
        population_sorted = [self.population[i] for i in sorted_indices] 
        scores_sorted = [self.scores[i] for i in sorted_indices] 
        
        self.bestPilots = population_sorted[:int(POPULATION * SURVIVAL_RATE)] # take the 10% bests pilots
        self.bestscores = scores_sorted[:int(POPULATION * SURVIVAL_RATE)]  # take the 10% bests scores
                
                
                
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
        
            
            




