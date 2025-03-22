import numpy as np
import math
#import matplotlib
#matplotlib.use('Qt5Agg')  # ou matplotlib.use('TkAgg') #problème de compatibilité avec pycharm, mettre avant import pyplot
import matplotlib.pyplot as plt



l = np.array([-0.615651859682504, -0.39348659288627863, 0.040000000000000036, 0.8999999999999999, -1.0, -1.0])

# print(l[:, np.newaxis])


best_scores = np.arange(0, 100, 1)
plt.plot(best_scores, label='Best scores')
#plt.plot(algoavg_scores, label='Average scores')
plt.xlabel("Générations")
plt.ylabel("Scores (%)")
plt.legend()
plt.show()