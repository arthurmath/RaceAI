import matplotlib.pyplot as plt
from pathlib import Path
import copy as cp
import random as rd
import pickle
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gene_pilot import Pilot
from gene_game import Session


SEED = 42
POPULATION = 500
SURVIVAL_RATE = 0.1
N_EPISODES = 100
N_STEPS = 80 
EPISODE_INCREASE = 2

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
        
        
        self.scores = [0] * 500

        for self.generation in range(N_EPISODES):
            
            self.evaluate_generation()        
            
            if self.ses.quit:
                break
            
            print(f"Generation {self.generation+1}, avg score: {self.avgGenScore:.2f}, best score: {self.bestGenScore:.2f}") # , mr: {self.mutation_rate:.2f}
            # print([round(x, 6) for x in self.scores[:50]])
            print()
            
        if not self.ses.quit:
            self.evaluate_generation() # Evaluate the last generation
            self.bests_survives()
            self.bestPilotEver = self.bestPilots[0]
            
            # print(f"BEST SCORE : {self.best_scores[0]:.3f}")
        


    def evaluate_generation(self):
            
        self.ses.reset(self.generation)
        states = self.ses.get_states()

        for step in range(N_STEPS + EPISODE_INCREASE * self.generation): # si pas d'augmentation du nb de steps, aucune différences entre les scores
            
            actions = [self.population[i].predict(states[i]) for i in range(len(self.population))]
        
            states = self.ses.step(actions)
            
            # print("STATES :", states)
            # print("ACTIONS : ", actions)
            
            if self.ses.done:
                break
            
        print(sum([1 if round(self.scores[i], 4) != round(self.ses.get_scores()[i], 4) else 0 for i in range(500)]))
        print(sum([1 if round(self.scores[i], 4) < round(self.ses.get_scores()[i], 4) else 0 for i in range(500)]))
        
        # for i in range(500):
        #     if round(self.scores[i], 4) != round(self.ses.get_scores()[i], 4):
        #         print(i, round(self.scores[i], 4), round(self.ses.get_scores()[i], 4))
        
            
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
        
        #self.new_list = [[self.bestPilots[i], self.best_scores[i]] for i in range(len(self.best_scores))]
        
        #print(self.new_list[:5])


                
    def change_generation(self):
        """ Creates a new generation of pilot. """
        
        self.new_population = cp.copy(self.bestPilots) # 10% best pilots
        #print(self.new_population[:5])

        # Check if the weights of the new population are the same as the best pilots
        print(all([all([(self.bestPilots[j].weights[i] == self.new_population[j].weights[i]).all() for i in range(3)]) for j in range(50)]))
        
        while len(self.new_population) < POPULATION:
            self.mutation_rate = max(1 - self.generation / MR_FACTOR, MR_MIN)
            
            if len(self.new_population) < 400:
                parent1, parent2 = self.select_parents_bests() # blue
                baby = parent1.mate(parent2)
                baby.mutate(0.3, std=0.1)
            else:
                baby = rd.choices(self.bestPilots[:5])[0] # green
                baby.mutate(0.1, std=0.1)
                
            self.new_population.append(baby)
        
        self.population = cp.copy(self.new_population)

        print(all([all([(self.bestPilots[j].weights[i] == self.population[j].weights[i]).all() for i in range(3)]) for j in range(50)]))
        print()
        
    
    def select_parents_bests(self):
        """Select two pilots with high scores among best ones."""
        total_scores = sum(self.best_scores)
        ratios = [f / total_scores for f in self.best_scores]
        return rd.choices(self.bestPilots, weights=ratios, k=2) # return a k-sized list 
    










if __name__ == "__main__":
    
    
    algo = GeneticAlgo()
    algo.train()
    