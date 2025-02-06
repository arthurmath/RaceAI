import multiprocessing as mp
import numpy as np
import time
import os
import timeit


"""MAP : applique la fonction à tous les éléments de la séquence et attend que toutes les tâches soient terminées 
avant de retourner les résultats sous forme d'une liste. C'est une opération synchrone, ce qui signifie 
que le programme reste bloqué jusqu'à ce que tous les résultats soient disponibles. """

"""IMAP : Retourne un "I"térable au lieu d'une liste complète (map). Les résultats sont produits 
 et peuvent être traités dès qu'ils sont disponibles, sans attendre que toutes les tâches soient terminées.
Imap est plus efficace pour traiter de grandes séquences. For very long iterables, using a large value 
for chunksize can make the job complete much faster than using the default value of 1. """



# Fonction	       	Ordre garanti ?	    Retour progressif des résultats?
# map()	           	    ✅ Oui       	   ❌ Non (list)
# map_async().get()	    ✅ Oui       	   ❌ Non (AsyncResult)
# imap_unordered()	    ❌ Non              ✅ Oui




## MAP
"""La méthode découpe l'itérable en morceaux (chunks) qu'elle envoie aux différents processus du pool 
 comme des tâches séparées. Peut entraîner une grosse consommation de mémoire pour les itérables très longs."""
 
def f(x):
    return x*x

if __name__ == '__main__':
    
    print("\nNumber of CPU:", mp.cpu_count(), "\n")
        
    with mp.Pool(processes=mp.cpu_count()) as pool:
        
        iter = list(range(10))
        
        # print(pool.map(f, iter))
        # print(pool.map_async(f, iter).get())
        # print([res for res in pool.imap_unordered(f, iter)])
        
        list_times = np.array(timeit.repeat('pool.map(f, iter)', globals=globals(), number=100, repeat=10)) / 1e2
        print(f"Execution time MAP : mean {(np.mean(list_times)/1e-3):.3f} ms, variance {(np.var(list_times)/1e-9):.2f} ns\n")
        
        list_times = np.array(timeit.repeat('pool.map_async(f, iter).get()', globals=globals(), number=100, repeat=10)) / 1e2
        print(f"Execution time MAP ASYNC : mean {(np.mean(list_times)/1e-3):.3f} ms, variance {(np.var(list_times)/1e-9):.2f} ns\n")
        
        list_times = np.array(timeit.repeat('[res for res in pool.imap_unordered(f, iter)]', globals=globals(), number=100, repeat=10)) / 1e2
        print(f"Execution time IMAP : mean {(np.mean(list_times)/1e-3):.3f} ms, variance {(np.var(list_times)/1e-9):.2f} ns\n")
        
        






# ## PROCESSUS
# """Un processus est une instance indépendante d'un programme en cours d'exécution, 
# avec son propre espace mémoire. Les processus sont isolés les uns des autres."""
# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#     print("")

# def func(name):
#     info('function f')
#     print('hello', name)

# if __name__ == '__main__':
#     info('main line')
#     p = mp.Process(target=func, args=('bob',))
#     p.start()
#     p.join(timeout=None)   
#     # Le processus parent attend que le processus enfant finisse son exécution avant de continuer.
#     # Si timeout est un nombre positif, elle bloque au maximum pendant timeout secondes.






### POOL 
# """ Un pool est un groupe de processus travailleurs (workers) 
# qui peuvent exécuter des tâches en parallèle. """
# def f(x):
#     return x*x

# if __name__ == '__main__':
#     try:
#         # start 4 worker processes
#         with mp.Pool(processes=4) as pool:

#             # print "[0, 1, 4,..., 81]"
#             print(pool.map(f, range(10)))

#             # print same numbers in arbitrary order
#             for res in pool.imap_unordered(f, range(10)):
#                 print(res)

#             # evaluate "f(20)" asynchronously
#             res = pool.apply_async(f, (20,))        # executed in one of the workers of the pool (apply_async juste pour 1 élem)
#             print(res.get(timeout=1))               # prints "400"

#             # launching multiple evaluations asynchronously, *may* use more processes
#             multiple_results = [pool.apply_async(os.getpid, ()) for _ in range(4)]
#             print([res.get(timeout=1) for res in multiple_results]) # prints the PID of that process

#             # make a single worker sleep for 10 seconds
#             res = pool.apply_async(time.sleep, (10,))
#             try:
#                 print(res.get(timeout=1)) # raise mp.TimeoutError if the result cannot be returned within timeout seconds.
#             except mp.TimeoutError:
#                 print("One second was too short!")

#             print("For the moment, the pool remains available for more work")

#         # exiting the 'with'-block has stopped the pool
#         print("Now the pool is closed and no longer available")
    
#     except KeyboardInterrupt:
#         # We have to stop the children
#         if mp.parent_process() is not None:
#             exit(107)











