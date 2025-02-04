from pilot import Pilot
from adn import Adn
from main import Session
import random as rd
import pickle
from pathlib import Path
from os import listdir



class SnakesManager:
    """
    Class whose purpose is to make the snakes plays and create new generations of snakes.
    """

    def __init__(self, ses, nbPilotes, layersSize=None, mutationRate=0.01, hunger=100, survivalProportion=0.1):
        """
        Constructor.
        ses (ses): Used to create a new game and GUI for a snake
        nbPilotes (int): The number of snake in each generation
        layersSize (list): A list containing the number of hidden neurons for each layer
        mutationRate (float): The probability for the DNA to mutate
        hunger (int): hunger (int): The starting hunger for each snake
        survivalProportion (float): The proportion of snakes that will replay in the next generation
        """
        self.ses = ses
        self.nbPilotes = nbPilotes
        self.layersSize = layersSize
        self.mutationRate = mutationRate
        self.hunger = hunger
        self.survivalProportion = survivalProportion

        self.snakes = [Pilot(Adn(layersSize=self.layersSize)) for i in range(nbPilotes)]
        # self.games = [TrainingSnakeGame(None) for i in range(nbPilotes)]
        
        self.generation = 0
        self.bestGenFitness = 0
        self.bestFitness = 0
        self.totalGenScore = 0
        self.bestSnake = self.snakes[0]  # The best snake of the generation

    def train(self):
        
        self.bestFitness = -1
        itEnd = 0

        # Create new generations until the stop condition is satisfied
        while itEnd < 150:
            print(f"Generation {self.generation}, best fitness: {self.bestFitness}")
            
            # Evaluation
            self.fitness = self.evaluate_gen()
            self.bestGenFitness = max(self.fitness)
            
            if self.bestGenFitness > self.bestFitness:
                self.bestFitness = self.bestGenFitness
                itEnd = 0
            else:
                # Si pas d'amélioration, le compteur aumgente
                itEnd += 1
                
            
            # Selection
            sorted_indices = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i])
            population_sorted = [self.population[i] for i in sorted_indices] 
            new_population = population_sorted[int(self.nbPilotes*9/10):] # take the 10% bests genomes


            # Croisement et mutations
            self.change_generation()

            


    def evaluate_gen(self):
        
        fitness = []
        for pilot in self.pilots:
            
            ses = Session(render=False, player=2, agent=pilot)
            ses.run()
            
            # Durée d'un run doit être finie
            
            self.fitness.append(pilot.compute_fitness(ses.car))
            
        return fitness
            
            

        # # Save the weights and biases of the snakes for the new game scores
        # files = listdir(Path("weights"))

        # # If this is a new game score
        # if str(self.bestGenScore) + ".snake" not in files:
        #     with open(Path("weights") / Path(str(self.bestGenScore) + ".snake"), "wb") as f: # write binary
        #         pickle.dump((self.bestSnake.dna.weights, self.bestSnake.dna.bias), f)
                
        
                
                
    def change_generation(self):
        """
        Creates a new generation of snakes.
        """
        # Sort the snakes by their fitness (decreasing)
        newSnakes = sorted(
            self.snakes, key=lambda x: x.get_fitness(), reverse=True
        )
        a1 = f"Average fitness for this generation: {sum([snake.get_fitness() for snake in self.snakes])/self.nbPilotes}"
        a2 = f"Median fitness for this generation: {newSnakes[len(newSnakes)//2].get_fitness()}"
        a3 = f"Best fitness for this generation: {self.bestGenFitness}"
        a4 = f"Best gamescore for this generation: {self.bestGenScore}"
        a5 = f"Average gamescore for this generation: {self.totalGenScore/self.nbPilotes}"

        print(f"{self.generation}\n{a1}\n{a2}\n{a3}\n{a4}\n{a5}\n\n")

        # Select best snakes
        newSnakes = newSnakes[: int(self.nbPilotes * self.survivalProportion)]

        # Generate new snakes
        while len(newSnakes) < self.nbPilotes:
            # Creates a new snake and add it to the next generation
            parents = self.pick_parents_rank(newSnakes)
            baby = parents[0].mate(parents[1], mutationRate=self.mutationRate)
            newSnakes.append(baby)

        # Update
        self.snakes = newSnakes
        self.bestGenFitness = 0
        self.bestGenScore = 0
        self.totalGenScore = 0
        self.generation += 1

    def pick_parents_rank(self, matingSnakes):
        """
        Pick two parents to mate and create a new snake that will participate in the next generation.
        The parents are selected according to their rank.
        """
        parents = []
        popSize = len(matingSnakes)
        totalFitness = popSize / 2 * (popSize + 1)

        for t in range(2):
            r = rd.randint(0, totalFitness - 1)
            i = 0
            used = None

            while i < len(matingSnakes):
                if r < popSize - i:
                    parents.append(matingSnakes[i])
                    totalFitness -= popSize
                    popSize -= 1
                    used = i
                    break

                r -= popSize - i
                i += 1

                if i == used:
                    i += 1

        return parents
    
    def show_best_snake(self):
        """
        Show the best snake of the current generation playing the game again.
        The snake replays in a new environment (new spawn position for the snake and food)
        """
        self.bestSnake.reset_state()
        self.ses.start_run()

        while self.ses.is_alive():
            self.ses.next_tick(self.bestSnake)




class TrainingSnakeGame(SnakeGame):
    def __init__(self, learning_agent):
        super(TrainingSnakeGame, self).__init__()
        self.learning_agent = learning_agent
        self.score = 0

    def next_tick(self):
        if self.is_alive():
            # print("Snake is alive, state: ", self.get_state())
            self.set_next_move(
                self.learning_agent.choose_next_move(self.get_state())
            )
            # print(self.next_move)
            if self.foodEaten:
                self.learning_agent.eat()
            return self.move_snake()

        return self.get_state()

    def get_score(self):
        return self.score








if __name__ == "__main__":
    
    ses = Session()
    
    population = 1000
    layers = [20, 10]
    mutation = 0.01
    hunger = 150
    elitism = 0.12
    snakesManager = SnakesManager(
        ses,
        population,
        layersSize=layers,
        mutationRate=mutation,
        hunger=hunger,
        survivalProportion=elitism,
    )
    
    snakesManager.train()