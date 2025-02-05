import numpy as np


class Pilot():
    """ Represents the AI that plays RaceAI """

    def __init__(self, adn): 
        self.adn = adn
        self.fitness = 0
        self.nbMove = 0
        self.actions = ['U', 'D', 'L', 'R']
        self.previous_moves = [0, 0]
        


    def choose_next_move(self, car):
        """ Choose a new move based on its state.
        Return the movement choice of the snake (tuple) """
        
        # Paramètres en entrée du réseau de neurones
        vision = [car.x, car.y, car.speed, car.angle, car.collision, car.nbCollisions, car.progression, *self.previous_moves] # * permet de déplier la liste
        
        # Il faut que les entrées soient dans [-1, 1] pour converger
        list_ranges = [[0, 1200], [0, 900], [-10, 10], [0, 360], [0, 1], [0, 500], [0, 100], [0, 3], [0, 3]]
        for idx, ranges in enumerate(list_ranges):
            vision[idx] = self.scale(vision[idx], *ranges)
            
        # Actions décidées par le réseau de neurones
        movesValues = self.adn.neural_network_forward(vision) 
        movesValues = movesValues.tolist()[0]  # listes de flottants dans [0, 1]
        
        # Choix des meilleures actions (celles avec une valeur > 0.7)
        choices = []
        for idx, x in enumerate(movesValues):
            if x > 0.7: # arbitraire
                choices.append(idx) # listes d'entiers dans [1, 4]

        self.previous_moves.extend(choices)
        while len(self.previous_moves) > 2:
            self.previous_moves.pop(0) # on ne garde que les 2 derniers moves
            
        self.nbMove += len(choices)
        
        self.moves = [self.actions[choice] for choice in choices]
            
        return self.moves
    
    
    
    def scale(self, x, a, b):
        """Transforme la valeur x initialement comprise dans l'intervalle [a, b]
            en une valeur comprise dans l'intervalle [-1, 1]."""
        return 2 * (x - a) / (b - a) - 1


    def compute_fitness(self, car):
        self.fitness = car.progression ** 2 / car.nbCollisions if car.nbCollisions else car.progression ** 2
        return self.fitness
    
            
    def mate(self, other, mutationRate):
        """ Mate with another pilot to create a new pilot """
        newDna = self.adn.mix(other.adn, mutationRate)
        return Pilot(newDna)
    

    def reset_state(self):
        self.nbMove = 0



