import numpy as np
import copy as cp
import random as rd


SEED = 42
NN_LAYERS = [5, 10, 10, 4]


np.random.seed(SEED)
rd.seed(SEED)



class Pilot:
   
    def __init__(self, weights=None, biases=None):
    
        if weights != None :
            self.weights = cp.deepcopy(weights)
        else:
            self.initialize_weights()
            
        if biases != None:
            self.bias = cp.deepcopy(biases)
        else:
            self.initialize_bias()
        
            
    def initialize_weights(self):
        self.weights = []
        for i in range(len(NN_LAYERS) - 1): 
            # Pour chaque couche du NN, creation d'une matrice de poids 
            layer = [[rd.uniform(-1, 1) for _ in range(NN_LAYERS[i+1])] for _ in range(NN_LAYERS[i])] # rd.gauss(0, 0.5)
            self.weights.append(np.matrix(layer))
            
        
    def initialize_bias(self):
        self.bias = []
        for layer in self.weights:
            nb_bias = layer.shape[1]
            self.bias.append(np.matrix([rd.uniform(-1, 1) for _ in range(nb_bias)])) # rd.gauss(0, 0.5)
            
    
    def predict(self, vector):
        for i, (weight, bias) in enumerate(zip(self.weights, self.bias)):
            vector = np.dot(np.array(vector), np.matrix(weight)) + np.array(bias)
            if i == len(self.weights):
                vector = self.relu(vector)
            else:
                vector = self.heaviside(vector)
        return vector
    
    def relu(self, x):
        return np.maximum(x, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def heaviside(self, x):
        return (x > 0).astype(int)
    
    



    # Pour l'entrainement
    
    def mate(self, other):
        """ Mix the copy of this DNA with the copy of another one to create a new one. """
        newWeights = self.crossover(self.weights, other.weights)
        newBias = self.crossover(self.bias, other.bias)
        return Pilot(newWeights, newBias)
    
    
    def crossover(self, dna1, dna2):
        """ Performs a crosover on the layers (weights and biases) """
        res = [self.cross_layer(dna1[layer], dna2[layer]) for layer in range(len(dna1))]
        return res

    def cross_layer(self, layer1, layer2): # better
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
    
    def cross_layer2(self, layer1, layer2):
        res = layer1
        if len(layer1.shape) == 1:  # 1D case
            for i in range(layer1.shape[0]):
                if rd.random() > 0.5:
                        res[i] = layer2[i]
        else:
            for i in range(layer1.shape[0]):
                for j in range(layer1.shape[1]):
                    if rd.random() > 0.5:
                        res[i, j] = layer2[i, j]
        return res
    

    def mutate(self, mutation_rate, std):
        for i, layer in enumerate(self.weights):
            self.weights[i] = self.mutate_layer(layer, mutation_rate, std)
        for i, layer in enumerate(self.bias):
            self.bias[i] = self.mutate_layer(layer, mutation_rate, std)
            

    def mutate_layer(self, layer, mutation_rate, std_mutation):
        """ Add a value from a gaussian distribution of mean 0 and standard deviation of std_mutation """
        
        mask = np.random.rand(*layer.shape) < mutation_rate # Tableau de True et False
        mutations = np.clip(np.random.normal(0, std_mutation, size=layer.shape), -1, 1) # -1 < mutations < 1 (stabilité numérique)
        layer = np.where(mask, layer + mutations, layer)  # condition, valeur_si_vrai, valeur_si_faux (layer += mask * mutations) 
        return np.matrix(layer)

    
    
    
    
    
    
    