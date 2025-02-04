import math


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
        movesValues = movesValues.tolist()[0]

        # Choix de la meilleure action (celle avec la + grande valeur)
        choice = movesValues.index(max(movesValues))
        
        # TODO à remplacer avec les moves > 80% ?
        # moves = []
        # for x in movesValues:
        #     if x > 0.8:
        #         moves.append(self.actions[movesValues.index(x)])

        self.previous_moves.append(choice)
        if len(self.previous_moves) > 2:
            self.previous_moves.pop(0)
            
        self.nbMove += 1
            
        return self.actions[choice]
    
    
    
    def scale(self, x, a, b):
        """Transforme la valeur x initialement comprise dans l'intervalle [a, b]
            en une valeur comprise dans l'intervalle [-1, 1]."""
        return 2 * (x - a) / (b - a) - 1



    def compute_fitness(self, car):
        
        self.fitness = car.progression ** 2 / car.nbCollisions
        return self.fitness
    
            
    def mate(self, other, mutationRate=0.01):
        """ Mate with another pilot to create a new pilot """
        newDna = self.adn.mix(other.dna, mutationRate)
        return Pilot(newDna)
    

    def reset_state(self):
        self.nbMove = 0
