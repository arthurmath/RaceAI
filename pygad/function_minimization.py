import pygad
import numpy as np


function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44


def fitness_function(ga_instance, solution, solution_idx):
    output = np.sum(solution * function_inputs)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness


ga_instance = pygad.GA(num_generations = 50,
                       num_parents_mating = 4,
                       fitness_func = fitness_function,
                       sol_per_pop = 8,
                       num_genes = len(function_inputs),
                       init_range_low = -2,
                       init_range_high = 5,
                       parent_selection_type = "sss",
                       keep_parents = 1,
                       crossover_type = "single_point",
                       mutation_type = "random",
                       mutation_percent_genes = 10)

ga_instance.run()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness:.3f}")

prediction = np.sum(np.array(function_inputs) * solution)
print(f"Predicted output based on the best solution : {prediction:.5f}")