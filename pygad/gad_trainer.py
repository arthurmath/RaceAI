import numpy as np
import pygad
import pygad.nn
import pygad.gann
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from gene_game import Session


SEED = 42
BATCH = 50
POPULATION = 2 * BATCH
SURVIVAL_RATE = 0.1
N_GENERATIONS = 20
N_STEPS = 100    
EPISODE_INCREASE = 2
NN_LAYERS = [5, 6, 6, 4]



    
    

def fitness_function(ga_instance, solution, nn_idx):
    """ Fonction de coût qui évalue la performance d'un réseau de neurones. """
    global gann
        
    ses = Session(display=True, nb_cars=len(nn_idx)) # 
    states = ses.get_states()
    
    generation = ga_instance.generations_completed

    for step in range(N_STEPS + EPISODE_INCREASE * generation):
        # print("GAN : ", len(gann.population_networks))
        # print("states:", len(states))
        # print("LEN : ", len(nn_idx))
        # print(nn_idx, "\n")
        actions = [pygad.nn.predict(last_layer=gann.population_networks[i], data_inputs=np.array(states)) for i in nn_idx]
        # print("actions:", len(actions))
        # actions = [[0]] * len(nn_idx)
        states = ses.step(actions)
        
        if ses.done:
            break
        
    scores = ses.get_scores()
        
    return scores



def callback(ga_instance):
    global gann
    population_matrices = pygad.gann.population_as_matrices(population_networks=gann.population_networks, population_vectors=ga_instance.population)
    gann.update_population_trained_weights(population_trained_weights=population_matrices)
    print(f"Generation: {ga_instance.generations_completed}, Best fitness: {ga_instance.best_solution()[1]}")





# Création de la population initiale
gann = pygad.gann.GANN(num_solutions=POPULATION,
                       num_neurons_input=NN_LAYERS[0],
                       num_neurons_hidden_layers=NN_LAYERS[1:-1],
                       num_neurons_output=NN_LAYERS[-1])

initial_pop = pygad.gann.population_as_vectors(population_networks=gann.population_networks)


# Définition des paramètres de l'algorithme génétique
ga_instance = pygad.GA(num_generations = N_GENERATIONS,
                    num_parents_mating = 10,
                    initial_population = initial_pop,
                    fitness_func = fitness_function,
                    mutation_percent_genes = 10,
                    crossover_type = 'single_point',
                    mutation_type = 'random',
                    parent_selection_type = 'tournament',
                    keep_parents = 2, 
                    init_range_high = 1,
                    init_range_low = -1,
                    random_seed = SEED,
                    fitness_batch_size = BATCH,
                    # parallel_processing = 5,
                    on_generation=callback,
                    )

# Exécution de l'algorithme génétique
ga_instance.run()

# Récupération du meilleur individu
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Meilleure solution: {solution}")
print(f"Fitness: {solution_fitness}")




# if __name__ == '__main__':
#     train()




# pourquoi nn_idx passe de len 32 à 31 au bout de 3 steps ? 
# Quel est le lien entre population et batch ? 
# Est ce que update_population_trained_weights est nécessaire ? 