import numpy as np
import time
import timeit
np.random.seed(42)


### Lignes Jean ###

# import matplotlib
# matplotlib.use('Qt5Agg')  # ou matplotlib.use('TkAgg') #problème de compatibilité avec pycharm, mettre avant import pyplot

###





size = 100
print("start\n")




a = np.random.rand(size, size)
b = np.random.rand(size, size)
list_times = np.array(timeit.repeat('np.dot(a, b)', globals=globals(), number=100, repeat=100)) / 1e2
print(f"Execution time Numpy : mean {(np.mean(list_times)/1e-3):.3f} ms, variance {(np.var(list_times)/1e-9):.2f} ns\n")





# JAX est une bibliothèque de Google qui accélère NumPy via GPU.
# !pip install jax jaxlib

import jax.numpy as jnp
import jax.random as jrd
key = jrd.PRNGKey(42)


a = jrd.uniform(key, shape=(size, size))
b = jrd.uniform(key, shape=(size, size))
list_times = np.array(timeit.repeat('jnp.dot(a, b)', globals=globals(), number=100, repeat=100)) / 1e2
print(f"Execution time JAX : mean {(np.mean(list_times)/1e-3):.3f} ms, variance {(np.var(list_times)/1e-9):.2f} ns\n")







# Incompatible avec MacOS 
# CuPy est une bibliothèque qui fonctionne comme NumPy mais qui exécute les opérations sur GPU via CUDA.
# !pip install cupy-cuda12x  # Remplace "12x" par ta version de CUDA

# import cupy as cp  # Remplace np par cp

# a = cp.random.rand(size, size)
# b = cp.random.rand(size, size)
# list_times = np.array(timeit.repeat('cp.dot(a, b)', globals=globals(), number=100, repeat=100)) / 1e2
# print(f"Execution time CUPY : mean {(np.mean(list_times)/1e-3):.3f} ms, variance {(np.var(list_times)/1e-9):.2f} ns\n")







# Il faut avoir Cuda installé
# Numba pour accélérer les boucles (JIT Compilation)

# from numba import cuda

# @cuda.jit
# def scalar_prod(a, b, c):
#     i = cuda.grid(1)
#     if i < a.size:  
#         c[i] = a[i] + b[i]

# @cuda.jit(nopython=True)
# def matrix_prod(A, B, C):
#     for i in range(size):
#         for j in range(size):
#             d = 0.0
#             for k in range(size):
#                 d += A[i,k] * B[k, j]    # Sans le coût de l'indexation de la matrice
#             C[i,j] = d
#     return C


# a = np.random.rand(size)
# b = np.random.rand(size)
# c = np.zeros(size)

# # Copier les données sur GPU
# d_a = cuda.to_device(a)
# d_b = cuda.to_device(b)
# d_c = cuda.to_device(c)

# # Lancer le kernel sur GPU
# threads_per_block = 256
# blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
# matrix_prod[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# # Récupérer les résultats en mémoire CPU
# c = d_c.copy_to_host()
# print(c[0, 0])










