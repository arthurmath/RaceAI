from pilot import Pilot
from adn import Adn
from main import Session
import random as rd
import pickle
from pathlib import Path
from os import listdir



class GeneticAlgo:
    """
    Class whose purpose is to make the snakes plays and create new generations of snakes.
    """

    def __init__(self, nbPilotes, mutation_rate, survival_rate, maxGenerations):
        """
        nbPilotes (int): The number of snake in each generation
        mutation_rate (float): The probability for the DNA to mutate
        survival_rate (float): The proportion of snakes that will replay in the next generation
        maxGenerations (int): Maximum number of generations
        """
        
        self.nbPilotes = nbPilotes
        self.mutationRate = mutation_rate
        self.survivalProportion = survival_rate
        self.maxGenerations = maxGenerations
        
    


    def train(self):
        
        self.bestFitness = -1
        self.bestGenFitness = -1
        self.bestScore = -1
        generation = 0
        itEnd = 0
        
        
        self.pilots = [Pilot(Adn()) for _ in range(self.nbPilotes)]

        # Create new generations until the stop condition is satisfied
        while itEnd < 150 or generation < self.maxGenerations:
            
            # Evaluation
            self.evaluate_generation()        
            
            # Selection
            self.bests_survives()

            # Reproduction
            self.change_generation()
            
            
            if self.bestGenFitness > self.bestFitness:
                itEnd = 0 # S'il y a une amélioration, on restart le compteur
            else:
                itEnd += 1 # Sinon, le compteur aumgente
            
            
            print(f"Generation {generation}, avg score: {self.avgGenScore}, best score: {self.bestGenScore}")
            generation += 1
            
        
        self.evaluate_generation()
        self.bests_survives()
        self.bestPilot = self.bestPilots[-1]
            

            


    def evaluate_generation(self):
        
        self.fitness = []
        self.scores = []
        
        for pilot in self.pilots: # multiproccessing TODO
            
            ses = Session(render=False, player=2, agent=pilot)
            ses.run()
            
            self.fitness.append(pilot.compute_fitness(ses.car))
            self.scores.append(ses.car.progression)
            
            
        self.bestGenFitness = max(self.fitness)
        self.bestGenScore = max(self.scores)
        self.avgGenScore = sum(self.scores) / self.nbPilotes
        
        if self.bestGenScore > self.bestScore:
            self.bestScore = self.bestGenScore
        
        if self.bestGenFitness > self.bestFitness:
            self.bestFitness = self.bestGenFitness
            
            
            
    def bests_survives(self):
        
        sorted_indices = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i])
        
        population_sorted = [self.population[i] for i in sorted_indices] 
        fitness_sorted = [self.fitness[i] for i in sorted_indices] 
        
        self.bestPilots = population_sorted[ int(self.nbPilotes * self.survivalProportion) :] # take the 10% bests pilots
        self.bestFitness = fitness_sorted[ int(self.nbPilotes * self.survivalProportion) :] # take the 10% bests fitness
                
                
                
    def change_generation(self):
        """ Creates a new generation of snakes. """
        
        self.new_population = self.bestPilots # 10% best pilots
        
        while len(self.new_population) < self.nbPilotes:
            parent1, parent2 = self.select_parents()
            baby = parent1.mate(parent2, self.mutationRate)
            self.new_population.append(baby)
                
        # Update
        self.population = self.new_population
        self.bestGenFitness = 0
        self.bestGenScore = 0
        

    
    def select_parents(self):
        """Select two parents with high fitness."""
        total_fitness = sum(self.bestFitness)
        ratios = [f / total_fitness for f in self.bestFitness]
        return rd.choices(self.new_population, weights=ratios, k=2)   # Return a k-sized list




    
    def show_best_snake(self):
        """
        Show the best snake of the current generation playing the game again.
        The snake replays in a new environment (new spawn position for the snake and food)
        """
        self.bestSnake.reset_state()
        self.ses.start_run()

        while self.ses.is_alive():
            self.ses.next_tick(self.bestSnake)






if __name__ == "__main__":
    
    population = 1000
    maxGenerations = 50 
    mutation_rate = 0.01
    survival_rate = 0.12
    
    
    algo = GeneticAlgo(population, mutation_rate, survival_rate, maxGenerations)
    algo.train()
    
    
    
    
    # Save the weights and biases of the snakes for the new game scores
    files = listdir(Path("weights"))
    
    with open(Path("weights") / Path(str(algo.bestGenScore) + ".weights"), "wb") as f: # write binary
        pickle.dump((algo.bestPilot.dna.weights, algo.bestPilot.dna.bias), f)
        
        
        