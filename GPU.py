

# !pip install cupy-cuda12x  # Remplace "12x" par ta version de CUDA

import cupy as cp  # Remplace np par cp

# Création de matrices sur GPU
a = cp.array([[1, 2], [3, 4]])
b = cp.array([[5, 6], [7, 8]])

# Produit matriciel sur GPU
c = cp.dot(a, b)
print(c)  # Résultat stocké sur GPU





# !pip install numba 

from numba import cuda
import numpy as np

@cuda.jit
def add_arrays_gpu(a, b, c):
    i = cuda.grid(1)
    if i < a.size:  
        c[i] = a[i] + b[i]

# Données en mémoire CPU
size = 1000000
a = np.random.rand(size)
b = np.random.rand(size)
c = np.zeros(size)

# Copier les données sur GPU
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

# Lancer le kernel sur GPU
threads_per_block = 256
blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
add_arrays_gpu[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Récupérer les résultats en mémoire CPU
c = d_c.copy_to_host()
print(c[:10])  # Affiche quelques résultats






# !pip install jax jaxlib


import jax.numpy as jnp

a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

# Opérations matricielles sur GPU
c = jnp.dot(a, b)
print(c)