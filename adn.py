import numpy as np
import copy as cp
import random as rd
rd.seed(42)



class Adn:
    
    def __init__(self, weights=None, biases=None):
        
        self.layersSize = [15, 10]
        # self.layersSize = [rd.randint(10, 15) for i in range(rd.randint(1, 3))]
            
        self.layersSize.insert(0, 9)  # Number of input neurons
        self.layersSize.append(4)  # Number of output neurons
        
        if weights is not None :
            self.weights = cp.deepcopy(weights)
        else:
            self.initialize_rd_weights()
            
        if biases is not None:
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
            vector = self.sigmoid(vector)  # Activation function
        return vector
    
    
    def relu(self, x):
        return np.maximum(x, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))



    # Pour l'entrainement
    
    def mix(self, other, mutationRate=0.01):
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

    def mutate(self, mutationRate=0.01):
        """ Mutate the DNA """
        
        for layer in self.weights:
            self.mutate_layer(layer, mutationRate)

        for layer in self.bias:
            self.mutate_layer(layer, mutationRate)
            
    def mutate_layer(self, layer, mutationRate=0.01):
        """ Add a value from a gaussian distribution of mean 0 and standard deviation of 0.5 """
                    
        mask = np.random.rand(*layer.shape) < mutationRate # Tableau de True et False
        mutations = np.clip(np.random.normal(0, 0.5, size=layer.shape), -1, 1) # -1 < mutations < 1 (stabilité numérique)
        layer = np.where(mask, layer + mutations, layer)  # condition, valeur_si_vrai, valeur_si_faux (layer += mask * mutations)


    
    
    
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
