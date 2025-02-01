import numpy as np
import time
np.random.seed(42)

print("start")


size = 1000


start = time.time()
a = np.random.rand(size, size)
b = np.random.rand(size, size)
c = np.dot(a, b)
print(c[0, 0])
print("Temps ecoule Numpy :", time.time() - start, "\n")






# JAX est une bibliothèque de Google qui accélère NumPy via GPU.
# !pip install jax jaxlib

import jax.numpy as jnp
import jax.random as jrd
key = jrd.PRNGKey(42)


start = time.time()
a = jrd.uniform(key, shape=(size, size))
b = jrd.uniform(key, shape=(size, size))
c = jnp.dot(a, b)
print(c[0, 0])
print("Temps ecoule JAX :", time.time() - start, "\n")






# Incompatible avec MacOS 
# CuPy est une bibliothèque qui fonctionne comme NumPy mais qui exécute les opérations sur GPU via CUDA.
# !pip install cupy-cuda12x  # Remplace "12x" par ta version de CUDA

# import cupy as cp  # Remplace np par cp

# start = time.time()
# a = cp.random.rand(size, size)
# b = cp.random.rand(size, size)
# c = cp.dot(a, b)
# print(c[0, 0])
# print("Temps ecoule Cupy :", time.time() - start, "\n")





# Il faut avoir Cuda installé
# Numba pour accélérer les boucles (JIT Compilation)

# from numba import cuda

# @cuda.jit
# def add_arrays_gpu(a, b, c): # produit scalaire
#     i = cuda.grid(1)
#     if i < a.size:  
#         c[i] = a[i] + b[i]


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
# add_arrays_gpu[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# # Récupérer les résultats en mémoire CPU
# c = d_c.copy_to_host()
# print(c[0, 0])









