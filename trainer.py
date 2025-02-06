import random as rd
import numpy as np
import pickle
import multiprocessing as mp
from pilot import Pilot, Adn
from main import Session
from pathlib import Path
from os import listdir



class GeneticAlgo:

    def __init__(self, nbPilotes, maxGenerations, mutation_rate, survival_rate):
        """
        nbPilotes (int): The number of snake in each generation
        maxGenerations (int): Maximum number of generations
        mutation_rate (float): The probability for the DNA to mutate
        survival_rate (float): The proportion of snakes that will replay in the next generation
        """
        
        self.nbPilotes = nbPilotes
        self.maxGenerations = maxGenerations
        self.mutationRate = mutation_rate
        self.survivalProportion = survival_rate
               
    

    def train(self):
        
        self.bestFit = - np.inf
        self.bestGenFit = - np.inf
        self.bestScore = - np.inf
        self.generation = 0
        itEnd = 0
        
        self.population = [Pilot(Adn()) for _ in range(self.nbPilotes)]

        # Create new generations until the stop condition is satisfied
        while itEnd < 50 and self.generation < self.maxGenerations:
            
            # Evaluation
            self.evaluate_generation_multi()        
            
            # Selection
            self.bests_survives()

            # Reproduction
            self.change_generation()
            
            
            if self.bestGenFit > self.bestFit:
                itEnd = 0 # S'il y a une amélioration, on restart le compteur
            else:
                itEnd += 1 # Sinon, le compteur aumgente
            
            print(f"\nGeneration {self.generation}, average progression: {self.avgGenScore:.3f}%, best progression: {self.bestGenScore:.3f}%\n")
            self.generation += 1
            
        
        self.evaluate_generation()
        self.bests_survives()
        self.bestPilotEver = self.bestPilots[-1]
        
        print(f"\nEnd of training, best progression achieved: {self.bestScore:.3f}%\n")



    def evaluate_generation(self):
        self.fitness = []
        self.scores = []
        
        for idx, pilot in enumerate(self.population): 
            
            ses = Session(train=True, agent=pilot, display=True, training_time=self.generation)
            ses.run()
            
            self.fitness.append(pilot.compute_fitness(ses.car))
            self.scores.append(ses.car.progression)
            
            print(f"Pilot {idx+1}/{self.nbPilotes}, fitness: {self.fitness[-1]:.3f}, progression: {self.scores[-1]:.3f}%")
        
        self.bestGenFit = max(self.fitness)
        self.bestGenScore = max(self.scores)
        self.avgGenScore = sum(self.scores) / self.nbPilotes
        
        if self.bestGenScore > self.bestScore:
            self.bestScore = self.bestGenScore
        
        if self.bestGenFit > self.bestFit:
            self.bestFit = self.bestGenFit
      
            
    def bests_survives(self):
        
        sorted_indices = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i])
        
        population_sorted = [self.population[i] for i in sorted_indices] 
        fitness_sorted = [self.fitness[i] for i in sorted_indices] 
        
        print(int(self.nbPilotes * self.survivalProportion), "\n")
        
        self.bestPilots = population_sorted[ int(self.nbPilotes * self.survivalProportion) :] # take the 10% bests pilots
        self.bestFitness = fitness_sorted[ int(self.nbPilotes * self.survivalProportion) :]  # take the 10% bests fitness
                
                
                
    def change_generation(self):
        """ Creates a new generation of snakes. """
        
        self.new_population = self.bestPilots # 10% best pilots
        
        while len(self.new_population) < self.nbPilotes:
            parent1, parent2 = self.select_parents()
            baby = parent1.mate(parent2, self.mutationRate)
            self.new_population.append(baby)
                
        # Update
        self.population = self.new_population
        
    
    def select_parents(self):
        """Select two pilots with high fitness."""
        total_fitness = sum(self.bestFitness)
        ratios = [f / total_fitness for f in self.bestFitness]
        return rd.choices(self.new_population, weights=ratios, k=2) # Return a k-sized list





    #### Calcul parallèle ####    
    
    def evaluate_generation_multi(self):
        cores = mp.cpu_count() # 8
        
        # Création d'un pool de processus (autant de processus que de coeurs)
        with mp.Pool(processes=cores) as pool:
            # Calcul de manière asynchrone (lancement de plusieurs processus distribués sur les coeurs)
            async_result = pool.map_async(self.run_pilot, self.population)
            results = async_result.get()
        
        
        # Récupération des listes de résultats 
        self.fitness, self.scores = map(list, zip(*results))
        
        # Update
        self.bestGenFit = max(self.fitness)
        self.bestGenScore = max(self.scores)
        self.avgGenScore = sum(self.scores) / self.nbPilotes
        
        if self.bestGenScore > self.bestScore:
            self.bestScore = self.bestGenScore
        
        if self.bestGenFit > self.bestFit:
            self.bestFit = self.bestGenFit
    

    def run_pilot(self, pilot): 
        """ Fonction exécutée dans un processus séparé pour chaque pilote. """
        
        ses = Session(train=True, agent=pilot, display=False, training_time=self.generation)
        ses.run()
        
        fitness = pilot.compute_fitness(ses.car)
        progression = ses.car.progression
        
        return fitness, progression
    
    







if __name__ == "__main__":
    
    population = 10 #100
    maxGenerations = 10 #50 
    mutation_rate = 0.01
    survival_rate = 0.1
    
    # Autres paramètres :
    # nombre de layers NN (adn)
    # parametre 0.7 pour la sélection des neurones (pilot)
    # temps d'entrainement de chaque pilote (main)
    # fps acceleration training
    
    
    algo = GeneticAlgo(population, maxGenerations, mutation_rate, survival_rate)
    algo.train()
    
    
    
    
    # Save the weights and biases of the snakes for the new game scores
    files = listdir(Path("weights"))
    
    with open(Path("weights") / Path(f"{algo.bestScore:.2f}.pilot"), "wb") as f: # write binary
        pickle.dump((algo.bestPilotEver.adn.weights, algo.bestPilotEver.adn.bias), f)
        
        
            
            






# comment empecher pyagem de print les truc dans lancement dans le terminal ?
