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


checkpoints = [(239, 273), (239, 130), (300, 75), (360, 130), (360, 392), (420, 451), (479, 389), (479, 126), 
                (531, 80), (941, 80), (988, 127), (988, 240), (940, 278), (680, 278), (614, 341), (681, 386), 
                (941, 386), (986, 440), (986, 750), (941, 800), (890, 800), (840, 751), (840, 583), (780, 532), 
                (680, 532), (620, 582), (620, 760), (570, 797), (238, 505), (238, 352)]


checkpoints2 = [(239, 273), (239, 130), (300, 75), (360, 130), (360, 392), (420, 451), (479, 389), (479, 126), 
               (531, 80), (941, 80), (988, 127), (988, 240), (940, 278), (680, 278), (614, 341), (681, 386), 
               (941, 386), (986, 440), (986, 750), (941, 800), (890, 800), (840, 751), (840, 583), (780, 532), 
               (680, 532), (620, 582), (620, 760), (570, 797), (301, 585), (236, 436)]

print(len(checkpoints))
print(len(checkpoints2))
print([i if checkpoints[i] != checkpoints2[i] else 0 for i in range(len(checkpoints))])