import math
import time
import numpy as np
import random as rd

rd.seed(42)


# In best_survives :

        # print([round(x, 2) for x in self.bestscores])
        # idx = 6
        # print(self.population[idx])
        # print(self.bestPilots[0])
        
        # import time
        # ses = Session(display=True, nb_cars=1)
        # ses.reset(self.generation)
        # states = ses.get_states()
        # for step in range(70):
        #     # actions = [self.bestPilots[0].predict(states).tolist()[0]]
        #     actions = [self.population[idx].predict(states).tolist()[0]]
        #     states = ses.step(actions)
        #     print(ses.get_scores()[0], ses.car_list[0].alive)
        #     time.sleep(0)
            
        
        
        # with open(Path("results_gene/weights") / Path(f"best.weights"), "wb") as f: # write binary
        #     pickle.dump((self.bestPilots[0].weights, self.bestPilots[0].bias), f)
        #     # pickle.dump((self.population[idx].weights, self.population[idx].bias), f)
        
        
        
        
# M = [[rd.uniform(-1, 1) for _ in range(3)] for _ in range(4)]
# M = np.array(M)
# print(type(M))

# M = np.random.random((4, 3))
# print(type(M))
        
# vector = [rd.uniform(-1, 1) for _ in range(4)]
# vector = np.array(vector)
            
    
# result = np.dot(np.array(vector), np.matrix(M))# + np.array(bias)
# result = np.dot(vector, M)
# print(result)




# NN_LAYERS = [5, 10, 10, 4]
# matrix = np.array([[rd.uniform(-1, 1) for _ in range(NN_LAYERS[1])] for _ in range(NN_LAYERS[0])])
# bias = np.array([rd.uniform(-1, 1) for _ in range(NN_LAYERS[1])])
# print(type(matrix))
# vector = np.array([1, 2, 3, 4, 5])
# res = vector @ matrix + bias
# print(type(res))



def distance_signed(p1, p2):
    return (p1[0] - p2[0]) + (p1[1] - p2[1])
    
p1 = (1, 2)
p2 = (3, 4)
print(p1[0])

print(distance_signed(p1, p2))