from pilot import Pilot
from adn import Adn
from main import Session
import random as rd
import numpy as np
import pickle
import multiprocessing as mp
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
        
        self.nbCPU = mp.cpu_count()
        print("Nombre de CPU :", self.nbCPU)
        
    

    def train(self):
        
        self.bestFit = - np.inf
        self.bestGenFit = - np.inf
        self.bestScore = - np.inf
        generation = 0
        itEnd = 0
        
        self.population = [Pilot(Adn()) for _ in range(self.nbPilotes)]

        # Create new generations until the stop condition is satisfied
        while itEnd < 50 or generation < self.maxGenerations:
            
            # Evaluation
            self.evaluate_generation_mono()        
            
            # Selection
            self.bests_survives()

            # Reproduction
            self.change_generation()
            
            
            if self.bestGenFit > self.bestFit:
                itEnd = 0 # S'il y a une amélioration, on restart le compteur
            else:
                itEnd += 1 # Sinon, le compteur aumgente
            
            
            print(f"\nGeneration {generation}, average progression: {self.avgGenScore:.3f}%, best progression: {self.bestGenScore:.3f}%\n")
            generation += 1
            
        
        self.evaluate_generation()
        self.bests_survives()
        self.bestPilotEver = self.bestPilots[-1]
        
        print(f"\nEnd of training, best progression achieved: {self.bestScore}%\n")



    def evaluate_generation_mono(self):
        self.fitness = []
        self.scores = []
        
        for idx, pilot in enumerate(self.population): 
            
            ses = Session(train=True, player=2, agent=pilot, display=True)
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




    # Calcul parallèle ####

    def run_pilot(self, arguments): # Cette fonction doit peut etre etre définie hors du __main__
        """ Fonction exécutée dans un processus séparé pour chaque pilote. """
        
        idx, pilot = arguments
        
        ses = Session(train=True, player=2, agent=pilot, display=True)
        ses.run()
        
        fitness = pilot.compute_fitness(ses.car)
        progression = ses.car.progression
        
        return idx, fitness, progression
    
    
    def evaluate_generation(self):
        
        arguments = [(idx, pilot) for idx, pilot in enumerate(self.population)]
        
        # Création d'un pool de processus
        with mp.Pool(processes=self.nbCPU) as pool:
            # Calcul de manière asynchrone (lancement de plusieurs processus distribués sur les coeurs)
            async_result = pool.map_async(self.run_pilot, arguments)
            
            # Attendre la fin de tous les processus
            async_result.wait()
            results = async_result.get()
            
        # Trier les résultats par index 
        results.sort(key=lambda x: x[0])
        
        # Récupération des résultats (sans index)
        _, self.fitness, self.scores = map(list, zip(*results))
        
        print(self.scores)
            
        # Update
        self.bestGenFit = max(self.fitness)
        self.bestGenScore = max(self.scores)
        self.avgGenScore = sum(self.scores) / self.nbPilotes
        
        if self.bestGenScore > self.bestScore:
            self.bestScore = self.bestGenScore
        
        if self.bestGenFit > self.bestFit:
            self.bestFit = self.bestGenFit
    
    
    







if __name__ == "__main__":
    
    population = 11 #100
    maxGenerations = 2 #50 
    mutation_rate = 0.01
    survival_rate = 0.1
    
    # Autres paramètres :
    # nombre de layers NN (adn)
    # parametre 0.7 pour la sélection des neurones (pilot)
    # temps d'entrainement de chaque pilote (main)
    
    
    algo = GeneticAlgo(population, maxGenerations, mutation_rate, survival_rate)
    algo.train()
    
    
    
    
    # Save the weights and biases of the snakes for the new game scores
    files = listdir(Path("weights"))
    
    with open(Path("weights") / Path(str(algo.bestGenScore) + ".pilot"), "wb") as f: # write binary
        pickle.dump((algo.bestPilotEver.adn.weights, algo.bestPilotEver.adn.bias), f)
        
        
            
            



# TODO enregistrement des weights et nouveau pilote a partir de ces weigths






