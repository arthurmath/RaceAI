import math


checkpoints = [[1, 1], [1, 2], [1, 3]] 


print(math.dist([1, 1], [1, 2]))

total_distance = 0
for i in range(len(checkpoints) - 1):
    print(checkpoints[i])
    total_distance += math.dist(checkpoints[i], checkpoints[i + 1])
    
print(total_distance)
    
    