import numpy as np
import copy as cp
import random as rd


SEED = 42
MUTATION_RATE = 0.9
STD_MUTATION = 1 #0.5
NN_LAYERS = [5, 6, 6, 4]


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
            nbrBias = np.size(layer, axis=1)
            self.bias.append(np.array([rd.uniform(-1, 1) for _ in range(nbrBias)])) # rd.gauss(0, 0.5)
            
    
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
        res = [ self.cross_layer(dna1[layer], dna2[layer]) for layer in range(len(dna1)) ]
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
            for i in range(len(layer1)):
                if rd.random() > 0.5:
                        res[i] = layer2[i]
        else:
            for i in range(len(layer1)):
                for j in range(len(layer1[0])):
                    if rd.random() > 0.5:
                        res[i][j] = layer2[i][j]
        return res
    

    def mutate(self):
        for layer in self.weights:
            self.mutate_layer(layer)
        for layer in self.bias:
            self.mutate_layer(layer)
            

    def mutate_layer(self, layer):
        """ Add a value from a gaussian distribution of mean 0 and standard deviation of STD_MUTATION """
        
        mask = np.random.rand(*layer.shape) < MUTATION_RATE # Tableau de True et False
        mutations = np.clip(np.random.normal(0, STD_MUTATION, size=layer.shape), -1, 1) # -1 < mutations < 1 (stabilité numérique)
        layer = np.where(mask, layer + mutations, layer)  # condition, valeur_si_vrai, valeur_si_faux (layer += mask * mutations) 


    
    
    
    
    
if __name__ == '__main__':
    
    state = [-1.0, -0.34, -0.73, -0.1, 0.8]

    pilot1 = Pilot()
    pilot2 = Pilot()

    action1 = pilot1.predict(state)
    print(action1.tolist()[0])
    
    action2 = pilot2.predict(state)
    print(action2.tolist()[0])
    
    
    
    
    
    
    
    
    
# predict()
# if weight.all() != self.weights[0].all():
#     vector = self.heaviside(vector)
# else: