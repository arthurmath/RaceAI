import numpy as np
import copy as cp
import random as rd

np.random.seed(42)
rd.seed(42)




class Pilot():
    """ Creates the AI Pilot that plays RaceAI """

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
        ### enlever nbCollisions (et collision ?)
        
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
            if x > 0.6: # arbitraire
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








class Adn:
    
    def __init__(self, weights=None, biases=None):
        
        self.layersSize = [15, 10]
        # self.layersSize = [rd.randint(10, 15) for i in range(rd.randint(1, 3))]
            
        self.layersSize.insert(0, 9)  # Number of input neurons
        self.layersSize.append(4)  # Number of output neurons
        
        if weights != None :
            self.weights = cp.deepcopy(weights)
        else:
            self.initialize_rd_weights()
            
        if biases != None:
            self.bias = cp.deepcopy(biases)
        else:
            self.initialize_rd_bias()
        
            
    def initialize_rd_weights(self):
        self.weights = []
        for i in range(len(self.layersSize) - 1): 
            # Creation des matrices contenant les poids pour chaque couche du NN
            layer = [[rd.gauss(0, 0.5) for _ in range(self.layersSize[i+1])] for _ in range(self.layersSize[i])]
            self.weights.append(np.matrix(layer))
            
        
    def initialize_rd_bias(self):
        self.bias = []
        for layer in self.weights:
            nbrBias = np.size(layer, axis=1)
            self.bias.append(np.array([rd.gauss(0, 0.5) for _ in range(nbrBias)]))
               
    
    
    def neural_network_forward(self, vector):
        for weight, bias in zip(self.weights, self.bias):
            vector = np.dot(np.array(vector), np.matrix(weight)) + np.array(bias)
            vector = self.relu(vector)  # Activation function
        return vector
    
    
    def relu(self, x):
        return np.maximum(x, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))



    # Pour l'entrainement
    
    def mix(self, other, mutationRate):
        """
        Mix the copy of this DNA with the copy of another one to create a new one.
        other (Dna): The other DNA used for the mixing
        mutationRate (float): The probability for a weight or bias to be mutated
        """
        newWeights = self.crossover(self.weights, other.weights)
        newBias = self.crossover(self.bias, other.bias)
        newDna = Adn(newWeights, newBias)
        newDna.mutate(mutationRate)
        return newDna
    
    
    def crossover(self, dna1, dna2):
        """ Performs a crosover on the layers (weights and biases) """
        res = [ self.cross_layer(dna1[layer], dna2[layer]) for layer in range(len(dna1)) ]
        return res

    def cross_layer(self, layer1, layer2):
        """ Performs a crossover on two layers """
        lineCut = rd.randint(0, layer1.shape[0] - 1)
        if len(layer1.shape) == 1:  # 1D case
            return np.hstack((layer1[:lineCut], layer2[lineCut:]))

        columnCut = rd.randint(0, layer1.shape[1] - 1)
        res = np.vstack((
            layer1[:lineCut],
            np.hstack((layer1[lineCut, :columnCut], layer2[lineCut, columnCut:])),
            layer2[lineCut + 1 :],
            ))
        return res
    

    def mutate(self, mutationRate):
        """ Mutate the DNA """
        
        for layer in self.weights:
            self.mutate_layer(layer, mutationRate)

        for layer in self.bias:
            self.mutate_layer(layer, mutationRate)
            
    def mutate_layer(self, layer, mutationRate):
        """ Add a value from a gaussian distribution of mean 0 and standard deviation of 0.5 """
                    
        mask = np.random.rand(*layer.shape) < mutationRate # Tableau de True et False
        mutations = np.clip(np.random.normal(0, 0.5, size=layer.shape), -1, 1) # -1 < mutations < 1 (stabilité numérique)
        layer = np.where(mask, layer + mutations, layer)  # condition, valeur_si_vrai, valeur_si_faux (layer += mask * mutations)


    
    
    
    
    
# TESTS 
    
# adn = Adn()

# print(adn.weights)
# print()
# print(adn.bias)
# print()

# inputs = np.array([rd.random() for _ in range(8)])
# print(adn.neural_network_forward(inputs))
# print()

# adn1 = Adn()
# adn2 = Adn()
# adn3 = adn1.mix(adn2)
# print(adn3.weights)
# adn.neural_network_forward(input)
