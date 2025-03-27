import math
import time
import numpy as np
import random as rd











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